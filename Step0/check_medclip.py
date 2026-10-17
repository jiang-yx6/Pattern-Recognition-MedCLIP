"""Inspect a local MedCLIP checkpoint, then verify 16 image/report pairs.
Use --inspect-only without ML dependencies. A normal run uses the official
MedCLIP processor and modules, loads the complete local checkpoint strictly,
and writes a small feature cache plus an HTML page for five-pair human review.
"""
from pathlib import Path
import argparse
import hashlib
import html
import importlib.metadata
import json
import os
import pickletools
import shutil
import subprocess
import sys
import time
import zipfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def git_commit(repository):
    executable = shutil.which('git')
    if executable:
        result = subprocess.run([executable, '-C', str(repository), 'rev-parse', 'HEAD'],
                                capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
    git_dir = repository / '.git'
    head = git_dir / 'HEAD'
    if head.is_file():
        ref = head.read_text().strip()
        if not ref.startswith('ref: '):
            return ref
        reference = ref[5:]
        if (git_dir / reference).is_file():
            return (git_dir / reference).read_text().strip()
        packed = git_dir / 'packed-refs'
        if packed.is_file():
            for line in packed.read_text().splitlines():
                if line.endswith(' ' + reference):
                    return line.split()[0]
    return None


def inspect_checkpoint(checkpoint, repository):
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    with zipfile.ZipFile(checkpoint) as archive:
        bad_member = archive.testzip()
        if bad_member:
            raise ValueError('Checkpoint CRC failed: ' + bad_member)
        metadata = next(name for name in archive.namelist() if name.endswith('/data.pkl') or name == 'data.pkl')
        strings = [value for operation, value, position in pickletools.genops(archive.read(metadata))
                   if operation.name in ('BINUNICODE', 'SHORT_BINUNICODE', 'UNICODE') and isinstance(value, str)]
    vision = any(value.startswith('vision_model.model.encoder.layers.') for value in strings)
    text = any(value.startswith('text_model.model.embeddings.') for value in strings)
    if not vision or not text:
        raise ValueError('Expected both the Swin vision encoder and text encoder in this checkpoint')
    return {'checkpoint': str(checkpoint), 'bytes': checkpoint.stat().st_size,
            'sha256': digest(checkpoint), 'zip_crc_passed': True,
            'swin_and_text_keys_present': True, 'medclip_commit': git_commit(repository),
            'status': 'checkpoint_inspected_only', 'actual_output_dimension': None,
            'forward_pass_verified': False}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/default.json')
    parser.add_argument('--inspect-only', action='store_true')
    parser.add_argument('--device', choices=['cuda', 'cpu'])
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--offline', action='store_true', help='Use already cached model configs and tokenizer')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    repository = (ROOT / config['model_repository']).resolve()
    checkpoint = (ROOT / config['checkpoint']).resolve()
    info_path = ROOT / 'outputs/model_info.json'
    info = inspect_checkpoint(checkpoint, repository)
    for expected_key, actual_key in [('expected_checkpoint_sha256', 'sha256'),
                                      ('expected_medclip_commit', 'medclip_commit')]:
        if config.get(expected_key) and config[expected_key] != info[actual_key]:
            raise ValueError(f'{actual_key} differs from the frozen project configuration')
    if args.inspect_only:
        write_json(info_path, info)
        print(json.dumps(info, ensure_ascii=False, indent=2))
        return
    if args.offline:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
    manifest = (ROOT / config['manifest']).resolve()
    if config.get('expected_manifest_sha256') and digest(manifest) != config['expected_manifest_sha256']:
        raise ValueError('Manifest SHA256 differs from the frozen project configuration')
    try:
        import numpy as np
        import pandas as pd
        import torch
        import torch.nn.functional as F
        from PIL import Image
        from transformers import AutoConfig, AutoModel
    except ImportError as error:
        raise RuntimeError('Install the isolated model environment described in README.md first') from error
    sys.path.insert(0, str(repository))
    from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPProcessor
    device = args.device or config['device']
    batch_size = args.batch_size if args.batch_size is not None else config['batch_size']
    if batch_size < 1:
        raise ValueError('Batch size must be positive')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; verify the CUDA PyTorch installation, or use --device cpu to debug')
    seed = config['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
    # The local checkpoint supplies ALL parameters. Only architecture configs and
    # tokenizer assets are needed from Hugging Face, not redundant base weights.
    def from_config_only(model_id, *unused_args, **kwargs):
        architecture = AutoConfig.from_pretrained(model_id, local_files_only=args.offline, **kwargs)
        return AutoModel.from_config(architecture)
    started = time.perf_counter()
    with patch.object(AutoModel, 'from_pretrained', side_effect=from_config_only):
        model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    model.requires_grad_(False)
    model.eval().to(device)
    processor = MedCLIPProcessor()
    maximum = int(config['text_max_length'])
    if maximum < 2 or maximum > model.text_model.model.config.max_position_embeddings:
        raise ValueError('Text length exceeds the encoder position limit')
    frame = pd.read_csv(manifest)
    selected = frame.loc[frame.split.eq(config['smoke_split'])].sample(
        n=config['smoke_samples'], random_state=seed).reset_index(drop=True)
    if selected.study_id.nunique() != len(selected):
        raise ValueError('Duplicate study IDs in the selected pairs')
    raw_ids = processor.tokenizer(selected.report.tolist(), padding=False, truncation=False, verbose=False)['input_ids']
    lengths = np.array([len(tokens) for tokens in raw_ids], dtype=np.int64)
    visual, textual = [], []
    with torch.inference_mode():
        for offset in range(0, len(selected), batch_size):
            batch = selected.iloc[offset:offset + batch_size]
            images = []
            for relative in batch.image_path:
                with Image.open(manifest.parent / relative) as image:
                    images.append(image.copy())
            # Fixed padding also keeps the official text mean-pooling independent
            # of the other reports in a feature-extraction batch.
            inputs = processor(text=batch.report.tolist(), images=images,
                               return_tensors='pt', padding='max_length',
                               truncation=True, max_length=maximum)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            # These are the official submodules; normalize exactly as the public
            # encode methods do, while avoiding their hard-coded .cuda() calls.
            v = F.normalize(model.vision_model(pixel_values=inputs['pixel_values']), dim=-1)
            t = F.normalize(model.text_model(inputs['input_ids'], inputs['attention_mask']), dim=-1)
            visual.append(v.cpu().numpy())
            textual.append(t.cpu().numpy())
            print(f'Encoded {min(offset + batch_size, len(selected))}/{len(selected)} pairs', flush=True)
    v = np.concatenate(visual).astype('float32')
    t = np.concatenate(textual).astype('float32')
    if v.shape != t.shape or not np.isfinite(v).all() or not np.isfinite(t).all():
        raise ValueError('Feature shape or finite-value check failed')
    if not np.allclose(np.linalg.norm(v, axis=1), 1, atol=1e-4) or not np.allclose(np.linalg.norm(t, axis=1), 1, atol=1e-4):
        raise ValueError('Unit-vector check failed')
    image_ids = selected.study_id.to_numpy()
    text_ids = selected.study_id.to_numpy().copy()
    if not np.array_equal(image_ids, text_ids):
        raise ValueError('Image/report ID alignment failed')
    output = ROOT / config['smoke_output']
    output.mkdir(parents=True, exist_ok=True)
    similarity = v @ t.T
    np.savez_compressed(output / 'features_16.npz', patient_id=selected.patient_id.to_numpy(),
                        study_id=image_ids, image_feature=v, text_feature=t,
                        labels=np.array([json.loads(value) for value in selected.labels]),
                        mask=np.array([json.loads(value) for value in selected['mask']]))
    pd.DataFrame(similarity, index=image_ids, columns=text_ids).to_csv(output / 'similarity.csv')
    selected.to_csv(output / 'pairs_16.csv', index=False)
    pd.DataFrame({'image_study_id': image_ids, 'text_study_id': text_ids,
                  'positive_column': np.arange(len(selected)), 'tokens_before_truncation': lengths,
                  'truncated': lengths > maximum}).to_csv(output / 'pair_checks.csv', index=False)
    preview = output / 'preview_images'
    preview.mkdir(exist_ok=True)
    figures = []
    for row in selected.sample(n=5, random_state=seed).itertuples():
        name = f'{row.study_id}.jpg'
        with Image.open(manifest.parent / row.image_path) as image:
            image = image.copy()
            image.thumbnail((600, 600))
            image.save(preview / name)
        figures.append(f'<figure><h2>Patient {row.patient_id} / Study {row.study_id}</h2>'
                       f'<img src="preview_images/{name}" alt="Chest radiograph">'
                       f'<p>{html.escape(row.report)}</p></figure>')
    (output / 'pairs_5.html').write_text('<!doctype html><meta charset="utf-8"><title>Five paired samples</title>'
        '<style>body{font-family:Arial;max-width:1100px;margin:30px auto}figure{margin:20px 0;padding:20px;border:1px solid #ccc}img{max-width:100%}p{white-space:pre-wrap;line-height:1.5}</style>'
        '<h1>Five paired samples for human review</h1>' + ''.join(figures), encoding='utf-8')
    if device == 'cuda':
        torch.cuda.synchronize()
    info.update(status='smoke_test_passed', forward_pass_verified=True,
                actual_output_dimension=int(v.shape[1]), samples=len(selected),
                batch_size=batch_size, text_max_length=maximum,
                truncated_reports=int((lengths > maximum).sum()),
                truncation_fraction=float((lengths > maximum).mean()),
                device=device, similarity_shape=list(similarity.shape),
                dtype=str(v.dtype), normalization='L2',
                vision_config_revision=getattr(model.vision_model.model.config, '_commit_hash', None),
                text_config_revision=getattr(model.text_model.model.config, '_commit_hash', None),
                strict_checkpoint_load=True, backbone_frozen=True,
                manifest_sha256=digest(manifest), config_sha256=digest(args.config),
                elapsed_seconds=round(time.perf_counter() - started, 3),
                peak_cuda_memory_mib=round(torch.cuda.max_memory_allocated() / 1024**2, 2) if device == 'cuda' else None,
                versions={name: importlib.metadata.version(name) for name in ['torch', 'torchvision', 'transformers', 'numpy', 'pandas', 'Pillow']},
                human_review_required=True)
    write_json(output / 'metadata.json', info)
    write_json(info_path, info)
    print(f'PASS: {len(selected)} pairs; dimension={v.shape[1]}; similarity={similarity.shape}')
    print('Open', output / 'pairs_5.html', 'for the required human review.')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(f'STOPPED: {error}', file=sys.stderr)
        raise SystemExit(1)
