
"""Metric heads, contrastive / hard-negative losses, retrieval metrics and training.

This module is the single home for everything behind the ``train_metric.py``
entry point, and also exposes the shared pieces used by the mining, retrieval
and geometry stages.

Public interface (imported by ``mining.py`` / ``retrieval.py`` / ``tests``)::

    MetricHead          shared (+ optional private) linear projection heads
    valid_negatives     which sample pairs are allowed to be negatives
    negative_weights    reweighting of hard negatives for methods B / D
    contrastive_loss    bidirectional InfoNCE, with optional hard weighting
    auxiliary_losses    reconstruction + shared/private covariance (C / D only)
    ranks / retrieval_values    fixed-library retrieval metrics
    load_head / transform_cache checkpoint loading and feature projection
    train_one           run one (method, seed) training
    main                CLI entry

Method matrix:

    A  plain in-batch negatives, one shared 256-d head per modality
    B  A + hard negatives: top 25% of valid negatives get weight 2
    C  A + shared/private disentanglement heads + reconstruction
    D  B + shared/private disentanglement heads + reconstruction
"""

import json

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

from .common import (
    Runtime,
    device_for,
    load_cache,
    load_config,
    output,
    parser,
    resolve,
    results,
    seed_all,
    sha256,
    write_json,
)


# ---------------------------------------------------------------------------
# 1. Metric heads
# ---------------------------------------------------------------------------


class MetricHead(nn.Module):
    """Linear projection heads built on top of frozen MedCLIP features.

    A/B: a single *shared* head per modality maps 512 -> 256, then L2-normalizes.
    C/D: additionally a *private* head per modality, plus a reconstruction head
    that maps the concatenated ``[shared, private]`` vector (2 * 256) back to the
    original feature dimension (512). Only the shared branch is used for retrieval.
    """

    def __init__(self, dimension, hidden, method):
        super().__init__()
        if method not in ['A', 'B', 'C', 'D']:
            raise ValueError('Metric heads are defined for A/B/C/D')
        self.method = method
        self.image_shared = nn.Linear(dimension, hidden)
        self.text_shared = nn.Linear(dimension, hidden)
        if method in ['C', 'D']:
            self.image_private = nn.Linear(dimension, hidden)
            self.text_private = nn.Linear(dimension, hidden)
            self.image_reconstruct = nn.Linear(2 * hidden, dimension)
            self.text_reconstruct = nn.Linear(2 * hidden, dimension)

    def forward(self, image, text):
        result = {
            'image_feature': F.normalize(self.image_shared(image), dim=-1),
            'text_feature': F.normalize(self.text_shared(text), dim=-1),
        }
        if self.method in ['C', 'D']:
            result['image_private'] = F.normalize(self.image_private(image), dim=-1)
            result['text_private'] = F.normalize(self.text_private(text), dim=-1)
            result['image_reconstruction'] = self.image_reconstruct(
                torch.cat([result['image_feature'], result['image_private']], dim=1))
            result['text_reconstruction'] = self.text_reconstruct(
                torch.cat([result['text_feature'], result['text_private']], dim=1))
        return result


# ---------------------------------------------------------------------------
# 2. Negative-sampling rules
# ---------------------------------------------------------------------------


def valid_negatives(labels, masks, patient_ids):
    """Return an N x N boolean matrix: ``valid[i, j]`` means j may be a negative for i.

    A pair is excluded when either:

    * the two samples belong to the same patient, or
    * both samples have a *non-empty* known-positive label set and those sets
      are identical.

    The second rule removes false negatives: two different studies describing the
    same disease(s) are semantically near-duplicates, so pushing them apart would
    teach the model the wrong thing. Unknown labels (mask == 0) never count as
    positive, so two "all unknown / all negative" samples keep an empty set and
    are still valid negatives of each other.
    """
    positive = (labels == 1) & (masks == 1)
    identical = (positive[:, None, :] == positive[None, :, :]).all(-1)
    nonempty = positive.any(-1)
    same_patient = patient_ids[:, None] == patient_ids[None, :]
    return ~(same_patient | (identical & nonempty[:, None] & nonempty[None, :]))


def negative_weights(similarity, valid, hard_fraction, hard_weight):
    """Up-weight the ``ceil(valid_count * hard_fraction)`` highest-similarity valid negatives.

    Returns ``(weights, hard)``:

    * ``weights`` is ``hard_weight`` on selected negatives, 1 everywhere else
      (diagonal / positive entries stay 1, matching the InfoNCE numerator).
    * ``hard`` is the boolean mask of the selected negatives.

    Selection is a stable descending ``argsort`` of the similarity over valid
    negatives only (invalid entries are masked to ``-inf``), so ties fall back to
    the earlier candidate position. ``scatter`` maps the rank-based mask back onto
    the original column positions, and the final ``& valid`` guarantees only valid
    negatives are ever weighted.
    """
    count = valid.sum(1)
    k = torch.ceil(count.float() * hard_fraction).long()
    ordered = torch.argsort(similarity.masked_fill(~valid, -torch.inf), dim=1,
                            descending=True, stable=True)
    hard_sorted = torch.arange(similarity.shape[1], device=similarity.device)[None, :] < k[:, None]
    hard = torch.zeros_like(valid).scatter(1, ordered, hard_sorted) & valid
    weights = torch.ones_like(similarity)
    weights[hard] = hard_weight
    return weights, hard


# ---------------------------------------------------------------------------
# 3. Losses
# ---------------------------------------------------------------------------


def contrastive_loss(image, text, labels, masks, patient_ids, temperature=0.07,
                     hard=False, fraction=0.25, weight=2.0, static_selected=None):
    """Bidirectional InfoNCE over a batch, with optional hard-negative weighting.

    The similarity matrix is ``image @ text.T``; row i is the image->text query i
    and column j is the text->image query j, so both directions reuse one matrix
    (``similarity`` and ``similarity.T``).

    Hard weighting is applied inside the log domain: multiplying a denominator
    term by ``w`` is the same as adding ``log(w)`` to its logit, so

        logits = similarity / temperature + log(weights)

    and the numerator stays ``similarity.diagonal() / temperature`` (weight 1).
    The diagonal is never masked out, even though it is also "invalid" as a
    negative, because it is the positive pair.

    ``static_selected``, when given, is a length-2 sequence of (n, n) boolean
    tensors (I2T then T2I) marking pre-selected hard-negative candidates that
    appear in the current batch. It replaces dynamic top-25% mining: only those
    pre-selected candidates that are also valid negatives get ``weight``.

    Returns ``(loss, skipped, hard_masks)`` where ``skipped`` counts the
    directions (rows) that had no valid negative and were dropped, and
    ``hard_masks`` holds the selected hard-negative mask per direction.
    """
    similarity = image @ text.T
    valid = valid_negatives(labels, masks, patient_ids)
    diagonal = torch.eye(len(image), dtype=torch.bool, device=image.device)

    losses = []
    hard_masks = []
    skipped = 0
    directions = [(similarity, valid), (similarity.T, valid.T)]
    for k, (scores, candidates) in enumerate(directions):
        active = candidates.any(1)
        skipped += int((~active).sum().item())
        if static_selected is not None:
            chosen = static_selected[k] & candidates
            weights = torch.ones_like(scores)
            weights[chosen] = weight
        elif hard:
            weights, chosen = negative_weights(scores, candidates, fraction, weight)
        else:
            weights = torch.ones_like(scores)
            chosen = torch.zeros_like(candidates, dtype=torch.bool)
        logits = (scores / temperature + weights.log()).masked_fill(
            ~(candidates | diagonal), -torch.inf)
        if active.any():
            losses.append(
                (torch.logsumexp(logits, dim=1) - scores.diagonal() / temperature)[active].mean())
        hard_masks.append(chosen)

    return (torch.stack(losses).mean() if losses else None), skipped, hard_masks


def auxiliary_losses(encoded, image, text):
    """Reconstruction and shared/private covariance losses for methods C / D.

    * Reconstruction MSE: mean over modalities of ``||reconstruct([shared, private]) - x||^2``.
    * Covariance: per modality, the mean squared entries of the cross-covariance
      matrix between the mean-centered shared and private vectors,
      ``(S - mean)^T (P - mean) / (n - 1)``.

    Returns ``(reconstruction, covariance)``, both zero when no private branch exists.
    """
    if 'image_private' not in encoded:
        zero = image.new_zeros(())
        return zero, zero

    reconstruction = (
        F.mse_loss(encoded['image_reconstruction'], image)
        + F.mse_loss(encoded['text_reconstruction'], text)
    ) / 2

    covariances = []
    for modality in ['image', 'text']:
        shared = encoded[modality + '_feature']
        private = encoded[modality + '_private']
        cross = (shared - shared.mean(0)).T @ (private - private.mean(0)) / max(len(shared) - 1, 1)
        covariances.append(cross.square().mean())

    return reconstruction, torch.stack(covariances).mean()


# ---------------------------------------------------------------------------
# 4. Retrieval metrics
# ---------------------------------------------------------------------------


def ranks(scores):
    """Rank of the diagonal element within each row, breaking ties by fixed order.

    ``scores`` must be a finite square matrix whose diagonal holds the single
    correct item per query. Equal scores are ordered by candidate position.
    """
    scores = np.asarray(scores)
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1] or not np.isfinite(scores).all():
        raise ValueError('Retrieval needs a finite square same-study similarity matrix')
    n = len(scores)
    truth = np.diag(scores)[:, None]
    ties = np.arange(n)[None, :] < np.arange(n)[:, None]
    return 1 + ((scores > truth) | ((scores == truth) & ties)).sum(1)


def retrieval_values(scores):
    """R@1/5/10 and mAP for both image->text and text->image directions."""
    result = {}
    for direction, matrix in [('I2T', scores), ('T2I', scores.T)]:
        r = ranks(matrix)
        result[direction] = {
            'R@1': float((r <= 1).mean()),
            'R@5': float((r <= 5).mean()),
            'R@10': float((r <= 10).mean()),
            'mAP': float((1 / r).mean()),
        }
    return result


# ---------------------------------------------------------------------------
# 5. Checkpoint and cache I/O
# ---------------------------------------------------------------------------


def run_dir(config, method, seed, smoke=False, hard_source='dynamic'):
    if smoke:
        return results(config, 'smoke_training', method, f'seed_{seed}')
    if hard_source == 'dynamic':
        return results(config, 'training', method, f'seed_{seed}')
    return results(config, 'training', method, f'seed_{seed}_{hard_source}')


def load_head(config, method, seed, device):
    """Load a trained metric head, verifying provenance against config and cache."""
    if method == 'R0':
        return None
    checkpoint = run_dir(config, method, seed) / 'model.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if (state['method'] != method or state['seed'] != seed
            or state['manifest_sha256'] != config['expected_manifest_sha256']):
        raise ValueError('Metric checkpoint provenance mismatch')
    if state['train_cache_sha256'] != sha256(output(config, 'features/train.npz')):
        raise ValueError('Training cache differs from the metric checkpoint')
    head = MetricHead(config['feature_dimension'], config['training']['head_dimension'], method)
    head.load_state_dict(state['state_dict'], strict=True)
    return head.eval().to(device)


def transform_cache(cache, head, device, prompts=1, branch='shared'):
    """Project cached features through ``head`` and return image/text embeddings.

    For the 5-prompt ensemble, each text variant is projected and L2-normalized
    first, then averaged, then normalized once more (never average raw features
    before the head). ``branch`` selects shared vs private heads for C / D.
    """
    if prompts not in [1, 5]:
        raise ValueError('Use one or five prompt variants')
    variants = cache['text_prompt_features']
    if variants.shape[0] < prompts:
        raise ValueError('Missing prompt variants: rerun extract_features for this split')

    with torch.inference_mode():
        image = torch.as_tensor(cache['image_feature'], device=device)
        texts = torch.as_tensor(variants[:prompts], device=device)
        if head is None:
            visual = image
            projected_texts = texts
        else:
            image_head = head.image_private if branch == 'private' else head.image_shared
            text_head = head.text_private if branch == 'private' else head.text_shared
            visual = F.normalize(image_head(image), dim=-1)
            projected_texts = F.normalize(text_head(texts), dim=-1)
        textual = F.normalize(projected_texts.mean(0), dim=-1)

    return visual.cpu().numpy(), textual.cpu().numpy()


def load_budget_candidates(config, method, seed, selection, study_ids):
    """Load pre-selected fixed-budget hard-negative candidates for one run.

    Reads the CSV written by ``mine_negatives.py --mode budget`` and returns
    ``{direction: {query_index: set(candidate_index)}}`` for one ``selection``
    strategy (``top20`` or ``conflict_first20``). Indices are positions in the
    train cache, aligned with ``study_ids``.
    """
    path = results(config, 'negatives', method, f'seed_{seed}', 'budget', 'negatives.csv')
    if not path.exists():
        raise FileNotFoundError(
            f'Run "mine_negatives.py --method {method} --seed {seed} --mode budget" first: {path}')
    frame = pd.read_csv(path)
    frame = frame.loc[frame.selection.eq(selection)]
    position = {int(study_id): i for i, study_id in enumerate(study_ids)}
    candidates = {'I2T': {}, 'T2I': {}}
    for _, row in frame.iterrows():
        query = position.get(int(row['query_id']))
        candidate = position.get(int(row['candidate_id']))
        if query is None or candidate is None:
            continue
        candidates[row['direction']].setdefault(query, set()).add(candidate)
    return candidates


# ---------------------------------------------------------------------------
# 6. Training
# ---------------------------------------------------------------------------


def _reusable_record(log_path, directory, config_sha, cache_sha):
    """Return the completed run's record if inputs still match, else raise.

    Returns ``None`` when no run exists yet; raises when a run exists but its
    config / training-cache fingerprint differs (caller must pass ``--force``).
    """
    if not log_path.exists():
        return None
    old = json.loads(log_path.read_text(encoding='utf-8'))
    matches = (
        old['config_sha256'] == config_sha
        and old['train_cache_sha256'] == cache_sha
        and (directory / 'model.pt').exists()
    )
    if matches:
        return old
    raise ValueError('Existing run uses different inputs; pass --force explicitly')


def _prepare_training(config, method, seed, train, val, device, smoke):
    """Seed RNG, build the head/optimizer and move the cached features to device."""
    seed_all(seed)
    settings = config['training']
    head = MetricHead(config['feature_dimension'], settings['head_dimension'], method).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=settings['learning_rate'], weight_decay=settings['weight_decay'])

    n = min(128, len(train['study_id'])) if smoke else len(train['study_id'])
    feature_keys = ['image_feature', 'text_feature', 'labels', 'mask', 'patient_id']
    tensors = {key: torch.as_tensor(train[key][:n], device=device) for key in feature_keys}
    val_image = torch.as_tensor(val['image_feature'], device=device)
    val_text = torch.as_tensor(val['text_feature'], device=device)
    rng = np.random.default_rng(seed)
    epochs = 2 if smoke else settings['max_epochs']
    return head, optimizer, tensors, val_image, val_text, rng, epochs, n


def _record_hard_examples(examples, hard_masks, encoded, idx, known_positive, train, mining, epoch):
    """Collect up to 20 image->text hard-negative pairs for manual inspection."""
    if len(examples) >= 20:
        return
    similarity = (encoded['image_feature'] @ encoded['text_feature'].T).detach().cpu().numpy()
    for row, col in hard_masks[0].nonzero().tolist():
        query = int(idx[row])
        candidate = int(idx[col])
        examples.append({
            'query_id': int(train['study_id'][query]),
            'candidate_id': int(train['study_id'][candidate]),
            'direction': 'I2T',
            'similarity': float(similarity[row, col]),
            'weight': mining['hard_weight'],
            'label_overlap': int((known_positive[query] & known_positive[candidate]).sum()),
            'epoch': epoch,
        })
        if len(examples) >= 20:
            return


def _static_selection(candidates, idx, device):
    """Boolean (n, n) matrix marking pre-selected candidates present in the batch.

    ``candidates`` maps train index -> set of train indices; ``idx`` maps batch
    position -> train index. ``out[i, j]`` is True when ``idx[j]`` is a
    pre-selected candidate of query ``idx[i]``.
    """
    n = len(idx)
    selected = np.zeros((n, n), dtype=bool)
    position = {int(t): i for i, t in enumerate(idx)}
    for i, query in enumerate(idx):
        for candidate in candidates.get(int(query), ()):
            j = position.get(int(candidate))
            if j is not None:
                selected[i, j] = True
    return torch.as_tensor(selected, device=device)


def _train_epoch(head, optimizer, tensors, order, settings, method, mining, train, epoch,
                 gradients, examples, static_candidates=None):
    """Run one epoch over the shuffled batch order.

    Mutates ``gradients`` (which parameters received non-zero gradient) and
    ``examples`` (hard-negative pairs for inspection). ``static_candidates``, when
    given, holds ``{direction: {query_index: set(candidate_index)}}`` and enables
    fixed-budget hard negatives instead of dynamic top-25% mining.

    Returns ``(records, skipped, hard_stats)`` where ``hard_stats`` aggregates
    ``selected`` (hard negatives actually weighted), ``overlap`` (their summed
    known-positive label overlap, a false-negative proxy) and ``activated`` (how
    many pre-selected candidates fell inside the batch).
    """
    head.train()
    hard = method in ('B', 'D') or static_candidates is not None
    known_positive = (train['labels'] == 1) & (train['mask'] == 1)

    records = []
    skipped = 0
    hard_stats = {'selected': 0, 'overlap': 0, 'activated': 0}
    for start in range(0, len(order), settings['batch_size']):
        idx = order[start:start + settings['batch_size']]
        if len(idx) < 2:
            skipped += 2 * len(idx)
            continue

        image = tensors['image_feature'][idx]
        text = tensors['text_feature'][idx]
        encoded = head(image, text)

        static_selected = None
        if static_candidates is not None:
            static_selected = [
                _static_selection(static_candidates['I2T'], idx, image.device),
                _static_selection(static_candidates['T2I'], idx, image.device),
            ]
            hard_stats['activated'] += int(static_selected[0].sum() + static_selected[1].sum())

        loss, skip, hard_masks = contrastive_loss(
            encoded['image_feature'], encoded['text_feature'],
            tensors['labels'][idx], tensors['mask'][idx], tensors['patient_id'][idx],
            settings['temperature'], hard, mining['hard_fraction'], mining['hard_weight'],
            static_selected,
        )
        skipped += skip
        if loss is None:
            continue

        reconstruction, covariance = auxiliary_losses(encoded, image, text)
        total = (
            loss
            + settings['reconstruction_weight'] * reconstruction
            + settings['covariance_weight'] * covariance
        )
        if not torch.isfinite(total):
            raise ValueError('Training loss is nonfinite')

        optimizer.zero_grad(set_to_none=True)
        total.backward()
        for name, parameter in head.named_parameters():
            if parameter.grad is None:
                continue
            if not torch.isfinite(parameter.grad).all():
                raise ValueError('Nonfinite gradient')
            gradients[name] |= bool(parameter.grad.abs().sum().item() > 0)
        optimizer.step()

        records.append([
            float(total.item()),
            float(loss.item()),
            float(reconstruction.item()),
            float(covariance.item()),
        ])
        if hard:
            for chosen in hard_masks:
                for row, col in chosen.nonzero().tolist():
                    query = int(idx[row])
                    candidate = int(idx[col])
                    hard_stats['selected'] += 1
                    hard_stats['overlap'] += int((known_positive[query] & known_positive[candidate]).sum())
            _record_hard_examples(examples, hard_masks, encoded, idx, known_positive,
                                  train, mining, epoch)

    return records, skipped, hard_stats


def _plot_losses(directory, history):
    """Save a 3-panel plot of the three loss components across epochs."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    for ax, name in zip(axes, ['contrastive_loss', 'reconstruction_loss', 'covariance_loss']):
        ax.plot([entry['epoch'] for entry in history], [entry[name] for entry in history])
        ax.set(xlabel='Epoch', ylabel=name)
    fig.tight_layout()
    fig.savefig(directory / 'losses.png')
    plt.close(fig)


def train_one(config, method, seed, train, val, device, smoke=False, force=False, hard_source='dynamic'):
    """Train one (method, seed) head, then write model/val-features/curves/record.

    Early stops when the mean of bidirectional val R@5 fails to improve for
    ``patience`` consecutive epochs (disabled for smoke runs). ``hard_source``
    selects ``dynamic`` (default, in-batch top-25% for B/D) or a fixed-budget
    strategy (``top20`` / ``conflict_first20``) whose candidates come from
    ``mine_negatives.py --mode budget``.
    """
    directory = run_dir(config, method, seed, smoke, hard_source)
    directory.mkdir(parents=True, exist_ok=True)
    config_sha = sha256(config['_config_path'])
    cache_sha = sha256(output(config, 'features/train.npz'))
    log_path = directory / 'runtime.json'

    if not force:
        old = _reusable_record(log_path, directory, config_sha, cache_sha)
        if old is not None:
            print(f'Reusing completed {method} seed {seed} ({hard_source})', flush=True)
            return old

    if hard_source == 'dynamic':
        static_candidates = None
    else:
        static_candidates = load_budget_candidates(config, method, seed, hard_source, train['study_id'])

    head, optimizer, tensors, val_image, val_text, rng, epochs, n = _prepare_training(
        config, method, seed, train, val, device, smoke)
    settings = config['training']
    mining = config['mining']

    best = -1.0
    best_epoch = 0
    stale = 0
    history = []
    initial = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    gradients = {name: False for name, _ in head.named_parameters()}
    examples = []
    total_skipped = 0
    hard_stats = {'selected': 0, 'overlap': 0, 'activated': 0}

    with Runtime(device) as runtime:
        for epoch in range(1, epochs + 1):
            order = rng.permutation(n)
            records, skipped, epoch_hard = _train_epoch(
                head, optimizer, tensors, order, settings, method, mining, train,
                epoch, gradients, examples, static_candidates)
            for key in hard_stats:
                hard_stats[key] += epoch_hard[key]

            if not records:
                raise ValueError('No query has valid negatives in this training run')

            head.eval()
            with torch.inference_mode():
                result = head(val_image, val_text)
                metrics = retrieval_values(
                    (result['image_feature'] @ result['text_feature'].T).cpu().numpy())
            score = (metrics['I2T']['R@5'] + metrics['T2I']['R@5']) / 2

            mean = np.mean(records, axis=0)
            history.append({
                'epoch': epoch,
                'total_loss': float(mean[0]),
                'contrastive_loss': float(mean[1]),
                'reconstruction_loss': float(mean[2]),
                'covariance_loss': float(mean[3]),
                'val_R5': score,
                'skipped_queries': skipped,
            })
            total_skipped += skipped
            print(f'{method} seed={seed} epoch={epoch}: loss={mean[0]:.5f} '
                  f'val_R5={score:.4f} skipped={skipped}', flush=True)

            if score > best:
                best, best_epoch, stale = score, epoch, 0
                torch.save({
                    'state_dict': {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                    'method': method, 'seed': seed,
                    'manifest_sha256': config['expected_manifest_sha256'],
                    'train_cache_sha256': cache_sha, 'config_sha256': config_sha,
                    'input_dimension': config['feature_dimension'],
                    'head_dimension': settings['head_dimension'], 'smoke_test': smoke,
                }, directory / 'model.pt')
            else:
                stale += 1
            if not smoke and stale >= settings['patience']:
                break

    changed = any(not torch.equal(initial[k], v.detach().cpu())
                  for k, v in head.state_dict().items())
    if not changed or not all(gradients.values()):
        raise ValueError('Some heads failed to receive gradients or update')

    head.load_state_dict(
        torch.load(directory / 'model.pt', map_location=device, weights_only=True)['state_dict'])
    head.eval()
    transformed = head(val_image, val_text)

    payload = {key: val[key] for key in ['patient_id', 'study_id', 'labels', 'mask']}
    payload.update(
        image_feature=transformed['image_feature'].detach().cpu().numpy(),
        text_feature=transformed['text_feature'].detach().cpu().numpy(),
        branch=np.array('shared' if method in ('C', 'D') else 'ordinary'),
    )
    if method in ('C', 'D'):
        payload.update(
            image_private=transformed['image_private'].detach().cpu().numpy(),
            text_private=transformed['text_private'].detach().cpu().numpy(),
        )
    np.savez_compressed(directory / 'val_features.npz', **payload)
    pd.DataFrame(history).to_csv(directory / 'losses.csv', index=False)
    if examples:
        pd.DataFrame(examples).to_csv(directory / 'hard_examples_20.csv', index=False)
    _plot_losses(directory, history)

    per_dimension_std = {
        key: np.std(value, axis=0).tolist()
        for key, value in payload.items()
        if key in ['image_feature', 'text_feature', 'image_private', 'text_private']
    }
    write_json(directory / 'per_dimension_std.json', per_dimension_std)

    record = {
        'method': method, 'seed': seed, 'hard_source': hard_source,
        'smoke_test': smoke, 'status': 'complete',
        'epochs': len(history), 'best_epoch': best_epoch, 'best_val_R5': best,
        'training_samples': n,
        'trainable_parameters': sum(x.numel() for x in head.parameters()),
        'head_parameters_updated': changed,
        'gradient_checks': gradients,
        'backbone_loaded': False, 'backbone_updated': False, 'backbone_frozen': True,
        'skipped_queries': total_skipped,
        'hard_negative_count': hard_stats['selected'],
        'hard_negative_label_overlap_rate': (hard_stats['overlap'] / hard_stats['selected']) if hard_stats['selected'] else 0.0,
        'static_activated_count': hard_stats['activated'],
        'checkpoint_sha256': config['expected_checkpoint_sha256'],
        'train_cache_sha256': cache_sha, 'config_sha256': config_sha,
        **runtime.record(),
    }
    write_json(log_path, record)
    return record


def main():
    p = parser('train_metric: cached-feature A/B/C/D training with seeds 42,43,44')
    p.add_argument('--method', choices=['ALL', 'A', 'B', 'C', 'D'], default='ALL')
    p.add_argument('--seed', type=int, choices=[42, 43, 44])
    p.add_argument('--device', choices=['cuda', 'cpu'])
    p.add_argument('--smoke-test', action='store_true',
                   help='128 train pairs, two epochs, separate output directory')
    p.add_argument('--force', action='store_true')
    p.add_argument('--hard-source', choices=['dynamic', 'top20', 'conflict_first20'], default='dynamic',
                   help='hard-negative source: dynamic in-batch top-25%% (B/D), or fixed-budget '
                        'top20 / conflict_first20 candidates from mine_negatives.py --mode budget (B only)')
    args = p.parse_args()
    if args.hard_source != 'dynamic' and args.method != 'B':
        p.error('--hard-source top20/conflict_first20 requires --method B')
    config = load_config(args.config)
    device = device_for(config, args.device)
    train = load_cache(config, 'train')
    val = load_cache(config, 'val')

    methods = [m for m in config['methods'] if m != 'R0'] if args.method == 'ALL' else [args.method]
    seeds = ([args.seed] if args.seed is not None
             else ([config['training']['seeds'][0]] if args.smoke_test
                   else config['training']['seeds']))

    before = sha256(resolve(config['checkpoint']))
    records = []
    for method in methods:
        for seed in seeds:
            records.append(train_one(config, method, seed, train, val, device,
                                     args.smoke_test, args.force, args.hard_source))
    if sha256(resolve(config['checkpoint'])) != before:
        raise ValueError('Frozen backbone checkpoint changed')

    name = 'smoke_training' if args.smoke_test else 'training'
    summary_suffix = '' if args.hard_source == 'dynamic' else f'_{args.hard_source}'
    write_json(results(config, name, f'run_summary{summary_suffix}.json'), records)
    summary_keys = ['method', 'seed', 'hard_source', 'epochs', 'best_epoch', 'best_val_R5',
                    'hard_negative_count', 'hard_negative_label_overlap_rate',
                    'elapsed_seconds', 'peak_cuda_memory_mib', 'trainable_parameters', 'smoke_test']
    summary = pd.DataFrame([{k: (r.get(k, 'dynamic') if k == 'hard_source' else r.get(k)) for k in summary_keys} for r in records])
    summary_path = results(config, 'logs', f'{name}{summary_suffix}.csv')
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    print('PASS:', len(records), 'training runs;', name, flush=True)
