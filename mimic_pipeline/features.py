import importlib.metadata
import json
import os
import sys
import time
from unittest.mock import patch
import numpy as np
import pandas as pd
from PIL import Image
from .common import Runtime, device_for, load_config, manifest_path, output, parser, resolve, seed_all, sha256, write_json


class Encoder:
    def __init__(self, c, device, offline=True):
        if offline:
            os.environ['HF_HUB_OFFLINE'] = '1'
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
        import torch
        from transformers import AutoConfig, AutoModel
        from .checkpoint import inspect_checkpoint
        self.info = inspect_checkpoint(resolve(c['checkpoint']), resolve(c['model_repository']))
        if self.info['sha256'] != c['expected_checkpoint_sha256'] or self.info['medclip_commit'] != c['expected_medclip_commit']:
            raise ValueError('Official code/checkpoint fingerprint mismatch')
        sys.path.insert(0, str(resolve(c['model_repository'])))
        from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPProcessor
        def config_only(name, *unused, **kwargs):
            return AutoModel.from_config(AutoConfig.from_pretrained(name, local_files_only=offline, **kwargs))
        with patch.object(AutoModel, 'from_pretrained', side_effect=config_only):
            model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
        state = torch.load(resolve(c['checkpoint']), map_location='cpu', weights_only=True)
        model.load_state_dict(state, strict=True)
        self.model = model.requires_grad_(False).eval().to(device)
        self.processor = MedCLIPProcessor()
        self.device = device; self.maximum = c['text_max_length']
        if not 2 <= self.maximum <= model.text_model.model.config.max_position_embeddings:
            raise ValueError('Invalid token length')
        self.info.update(vision_config_revision=getattr(model.vision_model.model.config, '_commit_hash', None),
                         text_config_revision=getattr(model.text_model.model.config, '_commit_hash', None))

    def text(self, texts):
        import torch
        import torch.nn.functional as F
        inputs = self.processor.tokenizer(texts, return_tensors='pt', padding='max_length', truncation=True, max_length=self.maximum)
        with torch.inference_mode():
            result = self.model.text_model(inputs['input_ids'].to(self.device), inputs['attention_mask'].to(self.device))
            return F.normalize(result, dim=-1).cpu().numpy().astype('float32')

    def images(self, images):
        import torch
        import torch.nn.functional as F
        inputs = self.processor(images=images, return_tensors='pt')
        with torch.inference_mode():
            return F.normalize(self.model.vision_model(pixel_values=inputs['pixel_values'].to(self.device)), dim=-1).cpu().numpy().astype('float32')


def main():
    p = parser('extract_features: frozen official MedCLIP features for all 5000 pairs')
    p.add_argument('--device', choices=['cuda','cpu'])
    p.add_argument('--batch-size', type=int)
    p.add_argument('--online', action='store_true', help='Allow Hugging Face downloads if cache is absent')
    p.add_argument('--force', action='store_true', help='Regenerate existing feature caches')
    args = p.parse_args(); c = load_config(args.config)
    seed_all(c['seed']); device = device_for(c, args.device)
    batch_size = args.batch_size or c['batch_size']
    if batch_size < 1:
        raise ValueError('Batch size must be positive')
    path = manifest_path(c)
    if sha256(path) != c['expected_manifest_sha256']:
        raise ValueError('Run prepare_data first; frozen manifest fingerprint differs')
    frame = pd.read_csv(path)
    base = output(c, 'features'); base.mkdir(parents=True, exist_ok=True)
    files = {}; stats = {}; prefix = c['features']['prompt_prefixes']
    with Runtime(device) as runtime:
        encoder = Encoder(c, device, offline=not args.online)
        for split, count in c['data']['split_counts'].items():
            part = frame.loc[frame.split.eq(split)].reset_index(drop=True)
            if len(part) != count:
                raise ValueError('Unexpected split count')
            destination = base / f'{split}.npz'; sidecar = base / f'{split}.json'
            prompt_count = len(prefix) if split in c['features']['prompt_splits'] else 1
            signature = {'manifest_sha256': sha256(path), 'checkpoint_sha256': encoder.info['sha256'],
                         'text_max_length': c['text_max_length'], 'prompt_prefixes': prefix[:prompt_count],
                         'medclip_commit': encoder.info['medclip_commit']}
            if destination.exists() and sidecar.exists() and not args.force:
                old = json.loads(sidecar.read_text(encoding='utf-8'))
                if old['signature'] != signature or old['sha256'] != sha256(destination):
                    raise ValueError('Stale/corrupt cache; use --force explicitly: ' + str(destination))
                with np.load(destination) as z:
                    if not np.array_equal(z['study_id'], part.study_id.to_numpy()):
                        raise ValueError('Existing cache ID order changed')
                files[split] = old; stats[split] = old['text_statistics']
                print('Verified and reused', destination, flush=True)
                continue
            token_ids = encoder.processor.tokenizer(part.report.tolist(), padding=False, truncation=False, verbose=False)['input_ids']
            lengths = np.array([len(x) for x in token_ids], dtype=np.int64)
            v = []; text = [[] for _ in range(prompt_count)]
            started = time.perf_counter()
            for start in range(0, len(part), batch_size):
                batch = part.iloc[start:start+batch_size]
                images = []
                for relative in batch.image_path:
                    image_path = (resolve(c['data']['image_root']) / relative).resolve()
                    image_path.relative_to(resolve(c['data']['image_root']) / 'images')
                    with Image.open(image_path) as image:
                        image.load(); images.append(image.copy())
                v.append(encoder.images(images))
                for j in range(prompt_count):
                    reports = [(prefix[j] + ' ' + r).strip() for r in batch.report]
                    text[j].append(encoder.text(reports))
                if start == 0 or (start + batch_size) % 100 == 0 or start+batch_size >= len(part):
                    print(f'{split}: {min(start+batch_size,len(part))}/{len(part)}', flush=True)
            visual = np.concatenate(v); variants = np.stack([np.concatenate(x) for x in text])
            for value in [visual, *variants]:
                if value.shape != (len(part), c['feature_dimension']) or not np.isfinite(value).all() or not np.allclose(np.linalg.norm(value, axis=1), 1, atol=1e-4):
                    raise ValueError('Nonfinite, nonunit or unexpected feature dimensions')
            payload = dict(patient_id=part.patient_id.to_numpy(dtype=np.int64), study_id=part.study_id.to_numpy(dtype=np.int64),
                           image_feature=visual, text_feature=variants[0], text_prompt_features=variants,
                           labels=np.array([json.loads(x) for x in part.labels], dtype=np.int64),
                           mask=np.array([json.loads(x) for x in part['mask']], dtype=np.int64),
                           tokens_before_truncation=lengths, branch=np.array('base'))
            temporary = destination.with_suffix('.tmp.npz')
            np.savez_compressed(temporary, **payload); temporary.replace(destination)
            stats[split] = {'reports': len(part), 'truncated': int((lengths > c['text_max_length']).sum()),
                            'truncation_fraction': float((lengths > c['text_max_length']).mean()),
                            'max_tokens': int(lengths.max()), 'median_tokens': float(np.median(lengths))}
            files[split] = {'sha256': sha256(destination), 'samples': len(part), 'dimension': visual.shape[1],
                            'signature': signature, 'elapsed_seconds': time.perf_counter()-started,
                            'text_statistics': stats[split], 'prompt_count': prompt_count}
            write_json(sidecar, files[split])
        prompts = c['mining']['disease_prompts']
        encoded = encoder.text([text for triple in prompts for text in triple]).reshape(5,3,c['feature_dimension'])
        import torch
        np.savez_compressed(base/'disease_prompts.npz', text_feature=encoded,
                            logit_scale=np.array(encoder.model.logit_scale.exp().clamp(max=100).item()),
                            prompts=np.array(prompts))
        if any(parameter.requires_grad for parameter in encoder.model.parameters()):
            raise ValueError('The backbone must remain frozen')
    metadata = {'status': 'complete', 'samples': len(frame), 'dimension': c['feature_dimension'],
                'manifest_sha256': sha256(path), 'checkpoint_sha256': encoder.info['sha256'],
                'medclip_commit': encoder.info['medclip_commit'], 'config_sha256': sha256(args.config),
                'vision_config_revision': encoder.info['vision_config_revision'], 'text_config_revision': encoder.info['text_config_revision'],
                'text_max_length': c['text_max_length'], 'padding': 'max_length', 'normalization': 'L2', 'dtype': 'float32',
                'strict_checkpoint_load': True, 'backbone_frozen': True, 'batch_size': batch_size, 'files': files,
                'processor': {'image_size': encoder.processor.feature_extractor.size,
                              'image_mean': encoder.processor.feature_extractor.image_mean,
                              'image_std': encoder.processor.feature_extractor.image_std,
                              'do_pad_square': encoder.processor.feature_extractor.do_pad_square},
                'versions': {name:importlib.metadata.version(name) for name in ['torch','torchvision','transformers','numpy','pandas','Pillow']},
                **runtime.record()}
    write_json(base/'metadata.json', metadata)
    total = sum(s['truncated'] for s in stats.values())
    write_json(output(c,'text_statistics.json'), {'max_length': c['text_max_length'], 'by_split':stats,
               'total_reports':len(frame), 'truncated':total, 'truncation_fraction':total/len(frame)})
    write_json(output(c,'logs/extract_features.json'), {'metadata_file':'features/metadata.json','samples':len(frame),**runtime.record()})
    print('PASS: 5000 pairs cached in', base, flush=True)
