"""Verify 16 pairs using the same frozen encoder as extract_features.py."""
from pathlib import Path
import html
import importlib.metadata
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from mimic_pipeline.common import Runtime, device_for, load_config, output, parser, resolve, seed_all, sha256, write_json
from mimic_pipeline.checkpoint import inspect_checkpoint


def main():
    p=parser(__doc__)
    p.add_argument('--inspect-only',action='store_true')
    p.add_argument('--device',choices=['cuda','cpu'])
    p.add_argument('--batch-size',type=int)
    p.add_argument('--offline',action='store_true')
    args=p.parse_args();c=load_config(args.config)
    if args.inspect_only:
        info=inspect_checkpoint(resolve(c['checkpoint']),resolve(c['model_repository']))
        if info['sha256']!=c['expected_checkpoint_sha256'] or info['medclip_commit']!=c['expected_medclip_commit']:
            raise ValueError('Official code/checkpoint fingerprint mismatch')
        write_json(output(c,'checks/checkpoint_inspection.json'),info)
        print('PASS: checkpoint inspection');return
    import json
    import numpy as np
    import pandas as pd
    from PIL import Image
    from mimic_pipeline.features import Encoder
    seed_all(c['seed']);device=device_for(c,args.device)
    batch_size=args.batch_size or c['batch_size']
    if batch_size<1:raise ValueError('Batch size must be positive')
    manifest=resolve(c['manifest'])
    if sha256(manifest)!=c['expected_manifest_sha256']:raise ValueError('Frozen manifest changed')
    frame=pd.read_csv(manifest)
    selected=frame.loc[frame.split.eq(c['smoke_split'])].sample(n=c['smoke_samples'],random_state=c['seed']).reset_index(drop=True)
    if selected.study_id.nunique()!=len(selected) or selected.patient_id.nunique()!=len(selected):
        raise ValueError('Duplicate patient or study IDs')
    image_root=resolve(c['data']['image_root'])
    visual=[];textual=[]
    with Runtime(device) as runtime:
        encoder=Encoder(c,device,offline=args.offline)
        ids=encoder.processor.tokenizer(selected.report.tolist(),padding=False,truncation=False,verbose=False)['input_ids']
        lengths=np.array([len(x) for x in ids],dtype=np.int64)
        for start in range(0,len(selected),batch_size):
            batch=selected.iloc[start:start+batch_size];images=[]
            for relative in batch.image_path:
                path=(image_root/relative).resolve();path.relative_to(image_root/'images')
                with Image.open(path) as image:images.append(image.copy())
            visual.append(encoder.images(images));textual.append(encoder.text(batch.report.tolist()))
            print(f'Encoded {min(start+batch_size,len(selected))}/{len(selected)} pairs',flush=True)
    v=np.concatenate(visual);t=np.concatenate(textual)
    for feature in [v,t]:
        if feature.shape!=(len(selected),c['feature_dimension']) or not np.isfinite(feature).all() or not np.allclose(np.linalg.norm(feature,axis=1),1,atol=1e-4):
            raise ValueError('Invalid feature dimensions, values or unit norms')
    folder=output(c,'checks/medclip_check');folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/'features_16.npz',patient_id=selected.patient_id.to_numpy(),study_id=selected.study_id.to_numpy(),
        image_feature=v,text_feature=t,labels=np.array([json.loads(x) for x in selected.labels]),mask=np.array([json.loads(x) for x in selected['mask']]))
    similarity=v@t.T
    pd.DataFrame(similarity,index=selected.study_id,columns=selected.study_id).to_csv(folder/'similarity.csv')
    selected.to_csv(folder/'pairs_16.csv',index=False)
    pd.DataFrame({'image_study_id':selected.study_id,'text_study_id':selected.study_id,'positive_column':np.arange(len(selected)),
                  'tokens_before_truncation':lengths,'truncated':lengths>c['text_max_length']}).to_csv(folder/'pair_checks.csv',index=False)
    preview=folder/'preview_images';preview.mkdir(exist_ok=True);figures=[]
    for row in selected.sample(n=5,random_state=c['seed']).itertuples():
        with Image.open(image_root/row.image_path) as source:
            image=source.copy();image.thumbnail((600,600));image.save(preview/f'{row.study_id}.jpg')
        figures.append(f'<figure><h2>Patient {row.patient_id} / Study {row.study_id}</h2><img src="preview_images/{row.study_id}.jpg"><p>{html.escape(row.report)}</p></figure>')
    (folder/'pairs_5.html').write_text('<!doctype html><meta charset="utf-8"><title>Five paired samples</title><style>body{font-family:Arial;max-width:1100px;margin:30px auto}figure{margin:20px 0;padding:20px;border:1px solid #ccc}img{max-width:100%}p{white-space:pre-wrap;line-height:1.5}</style>'+''.join(figures),encoding='utf-8')
    info={**encoder.info,'status':'smoke_test_passed','forward_pass_verified':True,'actual_output_dimension':int(v.shape[1]),
          'samples':len(selected),'batch_size':batch_size,'text_max_length':c['text_max_length'],
          'truncated_reports':int((lengths>c['text_max_length']).sum()),'truncation_fraction':float((lengths>c['text_max_length']).mean()),
          'similarity_shape':list(similarity.shape),'dtype':'float32','normalization':'L2','strict_checkpoint_load':True,'backbone_frozen':True,
          'manifest_sha256':sha256(manifest),'config_sha256':sha256(args.config),'human_review_required':True,
          'versions':{name:importlib.metadata.version(name) for name in ['torch','torchvision','transformers','numpy','pandas','Pillow']},**runtime.record()}
    write_json(folder/'metadata.json',info)
    print(f'PASS: {len(selected)} pairs; dimension={v.shape[1]}; similarity={similarity.shape}')
    print('Open',folder/'pairs_5.html')


if __name__=='__main__':
    try:main()
    except Exception as error:
        print(f'STOPPED: {error}',file=sys.stderr)
        raise SystemExit(1)
