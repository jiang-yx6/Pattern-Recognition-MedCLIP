from pathlib import Path
import argparse
import hashlib
import json
import os
import random
import time

ROOT = Path(__file__).resolve().parents[1]
LABELS = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--config', type=Path, default=ROOT / 'configs/default.json')
    return p


def load_config(path):
    c = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if c['methods'] != ['R0', 'A', 'B', 'C', 'D']:
        raise ValueError('The shared methods must be R0/A/B/C/D')
    if c['training']['seeds'] != [42, 43, 44] or c['seed'] != 42:
        raise ValueError('Keep split seed 42 and training seeds 42/43/44 fixed')
    c['_config_path'] = str(Path(path).resolve())
    cache = ROOT / '.cache' / 'matplotlib'
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(cache))
    os.environ.setdefault('MPLBACKEND', 'Agg')
    return c


def resolve(path):
    return (ROOT / path).resolve()


def output(c, *parts):
    return resolve(c['step0_dir']).joinpath(*parts)


def results(c, *parts):
    """Root for Step1 downstream outputs (training/mining/evaluation/geometry)."""
    return resolve(c['step1_dir']).joinpath(*parts)


def manifest_path(c):
    return output(c, 'indices', 'mimic_5000.csv')


def seed_all(seed):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)


def device_for(c, override=None):
    import torch
    device = override or c['device']
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; explicitly pass --device cpu for debugging')
    return torch.device(device)


class Runtime:
    def __init__(self, device=None):
        self.device = device

    def __enter__(self):
        if self.device is not None and self.device.type == 'cuda':
            import torch
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        self.started = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.peak_cuda_memory_mib = None
        if self.device is not None and self.device.type == 'cuda':
            import torch
            torch.cuda.synchronize(self.device)
            self.peak_cuda_memory_mib = torch.cuda.max_memory_allocated(self.device) / 1024**2
        self.elapsed_seconds = time.perf_counter() - self.started

    def record(self):
        return {'elapsed_seconds': self.elapsed_seconds,
                'peak_cuda_memory_mib': self.peak_cuda_memory_mib,
                'device': str(self.device) if self.device is not None else 'cpu'}


def load_cache(c, split):
    import numpy as np
    import pandas as pd
    path = output(c, 'features', f'{split}.npz')
    with np.load(path, allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    frame = pd.read_csv(manifest_path(c))
    frame = frame.loc[frame.split.eq(split)].reset_index(drop=True)
    if not np.array_equal(a['study_id'], frame.study_id.to_numpy()) or not np.array_equal(a['patient_id'], frame.patient_id.to_numpy()):
        raise ValueError('Cache ID order differs from the split manifest: ' + str(path))
    n = len(frame)
    for name in ['image_feature', 'text_feature']:
        x = a[name]
        if x.shape != (n, c['feature_dimension']) or not np.isfinite(x).all() or not np.allclose(np.linalg.norm(x, axis=1), 1, atol=1e-4):
            raise ValueError('Invalid unit feature cache: ' + name)
    labels = np.asarray([json.loads(x) for x in frame.labels], dtype=np.int64)
    masks = np.asarray([json.loads(x) for x in frame['mask']], dtype=np.int64)
    if not np.array_equal(a['labels'], labels) or not np.array_equal(a['mask'], masks):
        raise ValueError('Cache labels/mask differ from the manifest')
    meta = json.loads(output(c, 'features', 'metadata.json').read_text(encoding='utf-8'))
    if (meta['manifest_sha256'] != sha256(manifest_path(c)) or meta['checkpoint_sha256'] != c['expected_checkpoint_sha256']
            or meta['medclip_commit'] != c['expected_medclip_commit'] or meta['text_max_length'] != c['text_max_length']):
        raise ValueError('Feature provenance differs from the configured dataset/model')
    if meta['files'][split]['sha256'] != sha256(path):
        raise ValueError('Feature cache checksum changed: ' + str(path))
    return a
