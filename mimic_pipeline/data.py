import json
import re
import numpy as np
import pandas as pd
from .common import LABELS, ROOT, Runtime, load_config, output, parser, resolve, sha256, write_json

HEADINGS = 'FINDINGS|IMPRESSION|HISTORY|INDICATION|COMPARISON|TECHNIQUE|EXAMINATION|REASON FOR EXAMINATION|CLINICAL HISTORY'
PATTERNS = {name: re.compile(rf'(?ims)^\s*{name}\s*:\s*(.*?)(?=^\s*(?:{HEADINGS})\s*:|\Z)') for name in ['FINDINGS', 'IMPRESSION']}


def prepare(c):
    raw = resolve(c['data']['raw_dir'])
    frozen_path = resolve(c['data']['selection_file'])
    if sha256(frozen_path) != c['expected_selection_sha256']:
        raise ValueError('The frozen 5000-patient selection has changed')
    frozen = pd.read_csv(frozen_path)
    meta = pd.read_csv(raw / c['data']['metadata_file'])
    first = meta.groupby('subject_id').study_id.transform('min')
    eligible = meta.loc[meta.study_id.eq(first) & meta.ViewPosition.isin(['AP', 'PA'])]
    eligible = eligible.sort_values(['subject_id', 'study_id', 'dicom_id']).drop_duplicates('subject_id')
    chosen = frozen.merge(eligible, on=['subject_id', 'study_id', 'dicom_id'], how='left', sort=False, validate='one_to_one')
    if len(chosen) != sum(c['data']['split_counts'].values()) or chosen.ViewPosition.isna().any() or chosen.subject_id.nunique() != len(chosen):
        raise ValueError('Frozen selection fails the first-study/AP-PA rule')
    chex = pd.read_csv(raw / c['data']['chexpert_file'])
    chosen = chosen.merge(chex[['subject_id', 'study_id'] + LABELS], on=['subject_id', 'study_id'], how='left', sort=False, validate='many_to_one')
    ids = chosen.subject_id.to_numpy().copy()
    np.random.default_rng(c['seed']).shuffle(ids)
    ntrain = c['data']['split_counts']['train']; nval = c['data']['split_counts']['val']
    splits = {int(x): s for s, xs in [('train', ids[:ntrain]), ('val', ids[ntrain:ntrain+nval]), ('test', ids[ntrain+nval:])] for x in xs}
    rows = []; sections = []
    for r in chosen.itertuples():
        report_path = raw / 'reports/files' / f'p{str(r.subject_id)[:2]}' / f'p{r.subject_id}' / f's{r.study_id}.txt'
        text = report_path.read_text(encoding='utf-8')
        matches = [(name, m.group(1).strip()) for name, pattern in PATTERNS.items() if (m := pattern.search(text)) and m.group(1).strip()]
        if not matches:
            raise ValueError(f'No Findings/Impression: {report_path}')
        section, report = matches[0]
        sections.append({'subject_id': r.subject_id, 'study_id': r.study_id, 'section': section})
        # Pandas renames spaced itertuples fields; use the original column frame.
        values = chosen.loc[len(rows), LABELS].to_numpy(dtype=float)
        mask = np.isin(values, [0, 1]).astype(int)
        labels = np.where(mask, values, 0).astype(int)
        rel = f'p{str(r.subject_id)[:2]}/p{r.subject_id}/s{r.study_id}/{r.dicom_id}.jpg'
        rows.append({'subject_id': r.subject_id, 'study_id': r.study_id, 'dicom_id': r.dicom_id,
                     'ViewPosition': r.ViewPosition, 'relative_path': rel, 'image_path': 'images/' + rel,
                     'download_url': c['data']['url_root'] + rel, 'report': report, 'split': splits[int(r.subject_id)],
                     **dict(zip(LABELS, values)), 'labels': json.dumps(labels.tolist()), 'mask': json.dumps(mask.tolist())})
    full = pd.DataFrame(rows)
    main = full[['subject_id', 'study_id', 'image_path', 'report', 'split', 'labels', 'mask']].rename(columns={'subject_id': 'patient_id'})
    destination = output(c, 'indices'); destination.mkdir(parents=True, exist_ok=True)
    # Serialize before publishing and require the existing frozen manifest fingerprint.
    # Keep the frozen Windows CSV fingerprint reproducible on Linux/macOS too.
    main_bytes = main.to_csv(index=False, lineterminator='\r\n').encode('utf-8')
    import hashlib
    if hashlib.sha256(main_bytes).hexdigest() != c['expected_manifest_sha256']:
        raise ValueError('Reconstructed manifest differs from the frozen project manifest')
    (destination / 'mimic_5000.csv').write_bytes(main_bytes)
    full.to_csv(destination / 'mimic_5000_full.csv', index=False, lineterminator='\r\n')
    for split, count in c['data']['split_counts'].items():
        part = main.loc[main.split.eq(split)]
        if len(part) != count:
            raise ValueError('Wrong patient split size')
        part.to_csv(destination / f'{split}.csv', index=False, lineterminator='\r\n')
    frozen.to_csv(destination / 'patient_selection.csv', index=False, lineterminator='\r\n')
    write_json(destination / 'selection_provenance.json', {'seed': c['seed'], 'policy': 'fixed_subset_v1',
               'patient_selection_sha256': sha256(destination / 'patient_selection.csv'), 'patient_selection_file': (destination/'patient_selection.csv').relative_to(ROOT).as_posix(),
               'split_counts': c['data']['split_counts'], 'selection_rule': 'Smallest study_id before AP/PA; smallest dicom_id; Findings then Impression.'})
    write_json(destination / 'label_names.json', LABELS)
    train = full.loc[full.split.eq('train')]
    stats = [{'disease': name, 'positive_count': int(train[name].eq(1).sum()), 'negative_count': int(train[name].eq(0).sum()),
              'valid_count': int(train[name].isin([0,1]).sum()), 'uncertain_or_missing': int((~train[name].isin([0,1])).sum())} for name in LABELS]
    stats = pd.DataFrame(stats).sort_values('positive_count', ascending=False).reset_index(drop=True)
    stats.to_csv(destination / 'train_class_statistics.csv', index=False, lineterminator='\r\n')
    names = stats.disease.tolist()
    write_json(destination / 'frequency_groups.json', {'high_frequency': names[:2], 'middle_frequency': names[2:-2], 'low_frequency': names[-2:]})
    pd.DataFrame(sections).to_csv(destination / 'report_sections.csv', index=False)
    hypotheses=(ROOT/'hypotheses.md').read_text(encoding='utf-8').replace('subset/frequency_groups.json','indices/frequency_groups.json')
    output(c,'hypotheses.md').write_text(hypotheses,encoding='utf-8')
    snapshot = {k:v for k,v in c.items() if not k.startswith('_')}
    write_json(output(c, 'config_snapshot.json'), snapshot)
    output(c,'README.md').write_text('''# Step0 集中交付目录

- indices/：完整与分 split 索引、固定名单、标签顺序、类别频数、高低频分组、原始报告段落来源。
- features/：train/val/test 基础图文特征、提示特征、疾病提示、模型与文件指纹。图文基础维数均为 512。
- hypotheses.md：H1-H3 预先约定的比较组、主指标与预期方向；不是已证实的结论。
- config_snapshot.json：本次集中产物使用的公共配置。
- checks/：全量数据审计、16 对模型检查和 5 对图文核对页；检查元数据仅保留在各自目录。
- logs/：准备、提取和训练的运行记录与训练汇总 CSV。
- smoke_training/（可选）：仅 --smoke-test 生成的 128 对、2 轮调试结果，正式训练完成后可清理。
- training/：A-D 各种子的正式模型、损失曲线、梯度/冻结检查、耗时与峰值 CUDA 显存。
- negatives/：训练集负例、冲突清单及 20 对核对样本。
- evaluation/：固定候选库的检索指标、逐查询排名、均值/标准差、配对 bootstrap。
- geometry/：固定 val 的共享/私有几何指标、SVD 和联合 t-SNE。
- artifact_manifest.json：交付文件清单与 SHA256（在最终检查时生成，不包含其自身哈希）。

图片仍位于配置的 data.image_root/images；索引中的 image_path 相对于 data.image_root，不相对于本文件所在目录。
此目录含真实报告和逐样本数据，保留在本地，不因集中整理而改变原数据的访问和共享要求。
检查工具源码位于项目的 scripts/，交接说明位于 docs/handoff.md。
检索、挖掘和几何分析的耗时记录各自保存在结果目录的 runtime.json，不在 logs/ 重复汇总。
''',encoding='utf-8')
    return {'patients': len(main), 'split_counts': c['data']['split_counts'], 'manifest_sha256': sha256(destination/'mimic_5000.csv'),
            'source_selection_sha256': sha256(frozen_path), 'class_statistics': stats.to_dict('records')}


def main():
    p = parser('prepare_data: reconstruct the frozen split, statistics and hypotheses into the Step0 bundle')
    args = p.parse_args(); c = load_config(args.config)
    with Runtime() as runtime:
        result = prepare(c)
    write_json(output(c, 'logs/prepare_data.json'), {**result, **runtime.record()})
    print('PASS: prepared 5000 patients in', output(c))
