"""Read-only MIMIC subset audit. Writes audit artifacts only."""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
import argparse, hashlib, io, json, re, sys, time
import numpy as np
import pandas as pd
from PIL import Image, ImageFile

DISEASES = ['Atelectasis','Cardiomegaly','Consolidation','Edema','Pleural Effusion']
HEADINGS = 'FINDINGS|IMPRESSION|HISTORY|INDICATION|COMPARISON|TECHNIQUE|EXAMINATION|REASON FOR EXAMINATION|CLINICAL HISTORY'
PATTERNS = {s: re.compile(rf'(?ims)^\s*{s}\s*:\s*(.*?)(?=^\s*(?:{HEADINGS})\s*:|\Z)') for s in ['FINDINGS','IMPRESSION']}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--out',type=Path,default=None)
    parser.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[1]/'configs/default.json')
    args=parser.parse_args()
    root=args.root.resolve(); sub=root/'subset'
    config=json.loads(args.config.read_text(encoding='utf-8-sig'))
    out=args.out or root/config['step0_dir']/'checks'/'data_audit'
    out.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    checks=[]
    def check(name,condition,detail=None):
        checks.append({'check':name,'passed':bool(condition),'detail':detail})
    manifests=[sub/'mimic_5000.csv',sub/'mimic_5000_full.csv']
    hashes_before={p.name:sha(p) for p in manifests}
    df=pd.read_csv(manifests[0]); full=pd.read_csv(manifests[1])
    check('required_manifest_fields',set(['patient_id','study_id','image_path','report','split','labels','mask']).issubset(df.columns))
    check('5000_rows',len(df)==5000 and len(full)==5000)
    check('5000_unique_patients',df.patient_id.nunique()==5000)
    check('5000_unique_studies',df.study_id.nunique()==5000)
    check('5000_unique_image_paths',df.image_path.nunique()==5000)
    check('manifest_row_alignment',len(df)==len(full) and all(df[a].equals(full[b]) for a,b in [('patient_id','subject_id'),('study_id','study_id'),('image_path','image_path'),('report','report'),('split','split'),('labels','labels'),('mask','mask')]))
    check('no_empty_reports',df.report.notna().all() and df.report.fillna('').str.strip().ne('').all())
    split_counts={k:int(v) for k,v in df.split.value_counts().items()}
    check('split_4000_500_500',split_counts=={'train':4000,'val':500,'test':500},split_counts)
    sets={s:set(df.loc[df.split==s,'patient_id']) for s in ['train','val','test']}
    check('no_patient_leakage',not(sets['train']&sets['val'] or sets['train']&sets['test'] or sets['val']&sets['test']))
    ids=df.patient_id.to_numpy().copy(); np.random.default_rng(42).shuffle(ids)
    expected_split={int(v):s for s,arr in [('train',ids[:4000]),('val',ids[4000:4500]),('test',ids[4500:])] for v in arr}
    check('split_seed_42_reproduces',all(expected_split[int(r.patient_id)]==r.split for r in df.itertuples()))
    y=np.array([json.loads(x) for x in df.labels]); mask=np.array([json.loads(x) for x in df['mask']])
    check('labels_masks_shape_binary',y.shape==(5000,5) and mask.shape==(5000,5) and np.isin(y,[0,1]).all() and np.isin(mask,[0,1]).all())
    names=json.loads((sub/'label_names.json').read_text(encoding='utf-8'))
    check('fixed_label_order',names==DISEASES,names)
    meta=pd.read_csv(root/'raw/mimic-cxr-2.0.0-metadata.csv.gz')
    chex=pd.read_csv(root/'raw/mimic-cxr-2.0.0-chexpert.csv.gz')
    rawmeta=meta.set_index('dicom_id').reindex(full.dicom_id)
    check('metadata_join_ids_views',rawmeta.subject_id.notna().all() and np.array_equal(rawmeta.subject_id.to_numpy(),full.subject_id.to_numpy()) and np.array_equal(rawmeta.study_id.to_numpy(),full.study_id.to_numpy()) and np.array_equal(rawmeta.ViewPosition.to_numpy(),full.ViewPosition.to_numpy()))
    check('only_AP_PA',full.ViewPosition.isin(['AP','PA']).all())
    rawlabels=chex.set_index(['subject_id','study_id']).reindex(pd.MultiIndex.from_frame(full[['subject_id','study_id']]))[DISEASES].to_numpy()
    valid=np.isin(rawlabels,[0,1]); expected_y=np.where(valid,rawlabels,0).astype(int)
    check('labels_mask_match_original_chexpert',np.array_equal(y,expected_y) and np.array_equal(mask,valid))
    full_labels=full[DISEASES].to_numpy()
    check('full_index_original_labels_match',bool(np.all((full_labels==rawlabels)|(np.isnan(full_labels)&np.isnan(rawlabels)))))
    expected_paths=[]; expected_urls=[]
    for r in full.itertuples():
        rel=f'p{str(r.subject_id)[:2]}/p{r.subject_id}/s{r.study_id}/{r.dicom_id}.jpg'
        expected_paths.append('images/'+rel)
        expected_urls.append('https://physionet.org/files/mimic-cxr-jpg/2.1.0/files/'+rel)
    check('paths_match_ids',full.image_path.tolist()==expected_paths)
    check('download_urls_match_ids',full.download_url.tolist()==expected_urls)
    frontal=meta.loc[meta.ViewPosition.isin(['AP','PA'])].sort_values(['subject_id','study_id','dicom_id'])
    candidates=frontal.drop_duplicates('subject_id').reset_index(drop=True)
    chosen=candidates.set_index('subject_id').reindex(full.subject_id)
    check('first_frontal_study_and_first_dicom',np.array_equal(chosen.study_id.to_numpy(),full.study_id.to_numpy()) and np.array_equal(chosen.dicom_id.to_numpy(),full.dicom_id.to_numpy()))
    first_all=meta.groupby('subject_id').study_id.min()
    strict_min=full.subject_id.map(first_all)
    deviations=full.loc[full.study_id.ne(strict_min),['subject_id','study_id','dicom_id','ViewPosition','split']].copy()
    deviations['first_study_id_all_views']=strict_min.loc[deviations.index].to_numpy()
    frontal_studies=set(zip(frontal.subject_id,frontal.study_id))
    deviations['first_study_has_AP_PA']=[(int(r.subject_id),int(r.first_study_id_all_views)) in frontal_studies for r in deviations.itertuples()]
    check('strict_study_before_view_order',len(deviations)==0,{'deviations':len(deviations),'first_study_without_frontal':int((~deviations.first_study_has_AP_PA).sum())})
    if not deviations.empty:
        deviations.to_csv(out/'study_selection_deviations.csv',index=False,encoding='utf-8-sig')
    elif (out/'study_selection_deviations.csv').exists():
        (out/'study_selection_deviations.csv').unlink()
    report_cache={}; report_errors=[]
    def report(subject,study):
        key=(int(subject),int(study))
        if key in report_cache:return report_cache[key]
        p=root/'raw/reports/files'/f'p{str(subject)[:2]}'/f'p{subject}'/f's{study}.txt'
        try: text=p.read_text(encoding='utf-8')
        except (OSError,UnicodeError) as e:
            result=(None,None,str(e));report_cache[key]=result;return result
        for section,pattern in PATTERNS.items():
            found=pattern.search(text)
            if found and found.group(1).strip():
                result=(found.group(1).strip(),section,None); report_cache[key]=result;return result
        result=(None,None,'No nonempty Findings or Impression');report_cache[key]=result;return result
    sources=Counter();report_rows=[]
    for r in full.itertuples():
        body,section,error=report(r.subject_id,r.study_id)
        matches=body==r.report
        if not matches:report_errors.append({'subject_id':r.subject_id,'study_id':r.study_id,'error':error or 'Extracted text differs'})
        sources[section or 'MISSING']+=1
        report_rows.append({'subject_id':r.subject_id,'study_id':r.study_id,'section':section,'match':matches,'report_sha256':hashlib.sha256(r.report.encode()).hexdigest()})
    check('all_original_reports_found_and_sections_match',not report_errors,{'section_counts':dict(sources),'errors':report_errors[:10]})
    pd.DataFrame(report_rows).to_csv(out/'report_sources.csv',index=False,encoding='utf-8-sig')
    provenance=json.loads((sub/'selection_provenance.json').read_text(encoding='utf-8'))
    if provenance.get('policy')!='fixed_subset_v1':
        raise ValueError('Unsupported selection policy; expected fixed_subset_v1')
    selection_path=(root/provenance['patient_selection_file']).resolve()
    selection_path.relative_to(root)
    frozen=pd.read_csv(selection_path)
    got=list(full[['subject_id','study_id','dicom_id']].itertuples(index=False,name=None))
    expected=list(frozen[['subject_id','study_id','dicom_id']].itertuples(index=False,name=None))
    check('frozen_patient_selection_matches_manifest',
          got==expected and sha(selection_path)==provenance['patient_selection_sha256'] and provenance['seed']==42,
          {'policy':provenance['policy'],'patient_count':len(frozen)})
    stats=[]
    for j,name in enumerate(DISEASES):
        vals=rawlabels[df.split.eq('train').to_numpy(),j]
        stats.append({'disease':name,'positive_count':int((vals==1).sum()),'negative_count':int((vals==0).sum()),'valid_count':int(np.isin(vals,[0,1]).sum()),'uncertain_or_missing':int((~np.isin(vals,[0,1])).sum())})
    stats=pd.DataFrame(stats).sort_values('positive_count',ascending=False).reset_index(drop=True)
    check('saved_class_statistics_match',stats.equals(pd.read_csv(sub/'train_class_statistics.csv')))
    ordered=stats.disease.tolist(); groups={'high_frequency':ordered[:2],'middle_frequency':ordered[2:-2],'low_frequency':ordered[-2:]}
    check('frequency_groups_train_only_match',groups==json.loads((sub/'frequency_groups.json').read_text()),groups)
    print('INDEX_REPORT_CHECKS_DONE',len(checks),'Starting full image decode + SHA256',flush=True)
    ImageFile.LOAD_TRUNCATED_IMAGES=False
    image_rows=[];image_errors=[];hash_ids=defaultdict(list);dims=[];modes=Counter();totalbytes=0;zero_dynamic=[]
    for i,r in enumerate(full.itertuples()):
        p=sub/r.image_path
        result={'subject_id':int(r.subject_id),'study_id':int(r.study_id),'image_path':r.image_path,'split':r.split,'status':'ok'}
        try:
            p.resolve().relative_to((sub/'images').resolve())
            data=p.read_bytes();totalbytes+=len(data);result['bytes']=len(data)
            h=hashlib.sha256(data).hexdigest();result['sha256']=h;hash_ids[h].append(int(r.study_id))
            with Image.open(io.BytesIO(data)) as im: im.verify()
            with Image.open(io.BytesIO(data)) as im:
                im.load(); dims.append(im.size);modes[im.mode]+=1
                result.update({'width':im.width,'height':im.height,'mode':im.mode,'format':im.format})
                extrema=im.getextrema()
                if im.mode=='L' and extrema[0]==extrema[1]:zero_dynamic.append(int(r.study_id))
                result['matches_metadata_dimensions']=im.width==int(rawmeta.iloc[i]['Columns']) and im.height==int(rawmeta.iloc[i]['Rows'])
                if im.format!='JPEG':raise ValueError('Non-JPEG content')
        except Exception as e:
            result['status']='missing' if not p.exists() else 'error';result['error']=str(e);image_errors.append(result.copy())
        image_rows.append(result)
        if (i+1)%500==0:print(f'IMAGES {i+1}/5000 errors={len(image_errors)}',flush=True)
    pd.DataFrame(image_rows).to_csv(out/'image_integrity.csv',index=False,encoding='utf-8-sig')
    if image_errors:
        pd.DataFrame(image_errors).to_csv(out/'image_errors.csv',index=False,encoding='utf-8-sig')
    elif (out/'image_errors.csv').exists():
        (out/'image_errors.csv').unlink()
    disk_images={p.relative_to(sub).as_posix() for p in (sub/'images').rglob('*.jpg')}
    indexed=set(full.image_path)
    check('5000_images_present_and_fully_decodable',len(image_rows)==5000 and not image_errors,{'errors':len(image_errors)})
    check('images_match_metadata_dimensions',all(r.get('matches_metadata_dimensions',False) for r in image_rows))
    check('no_constant_grayscale_images',not zero_dynamic,zero_dynamic)
    check('on_disk_image_inventory_matches_manifest',disk_images==indexed,{'extra':sorted(disk_images-indexed),'missing':sorted(indexed-disk_images)})
    duplicate_image_groups=[v for v in hash_ids.values() if len(v)>1]
    check('no_exact_duplicate_image_files',not duplicate_image_groups,duplicate_image_groups)
    check('manifest_unchanged_during_audit',hashes_before=={p.name:sha(p) for p in manifests})
    timestamp=datetime.now().astimezone().isoformat()
    summary={'audit_timestamp_host':timestamp,'client_date':datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d'),'root':str(root),'checks':checks,'passed_checks':sum(c['passed'] for c in checks),'total_checks':len(checks),'failed_checks':[c for c in checks if not c['passed']],
        'sample_count':len(df),'patients':int(df.patient_id.nunique()),'split_counts':split_counts,'view_counts':{str(k):int(v) for k,v in full.ViewPosition.value_counts().items()},
        'images':{'present':sum(r['status']!='missing' for r in image_rows),'decode_passed':sum(r['status']=='ok' for r in image_rows),'missing':sum(r['status']=='missing' for r in image_rows),'decode_failed':sum(r['status']=='error' for r in image_rows),'bytes':totalbytes,'modes':dict(modes),'min_width_height':np.min(dims,axis=0).tolist() if dims else [],'max_width_height':np.max(dims,axis=0).tolist() if dims else [],'exact_duplicate_groups':duplicate_image_groups},
        'reports':{'section_counts':dict(sources),'missing_or_mismatched':len(report_errors),'duplicate_rows_beyond_first':int(df.report.duplicated().sum()),'cross_split_equal_text_counts':{a+'_'+b:len(set(df.loc[df.split==a,'report'])&set(df.loc[df.split==b,'report'])) for a,b in [('train','val'),('train','test'),('val','test')]}},
        'all_five_masks_zero':{s:int((mask[df.split.eq(s).to_numpy()].sum(1)==0).sum()) for s in ['train','val','test']},'class_statistics':stats.to_dict('records'),'frequency_groups':groups,
        'study_selection':{'strict_order_deviations':len(deviations),'deviations_by_split':{str(k):int(v) for k,v in deviations.split.value_counts().items()},'all_deviations_first_study_without_AP_PA':bool((~deviations.first_study_has_AP_PA).all())},'manifest_sha256':hashes_before,'elapsed_seconds':round(time.perf_counter()-started,2),
        'limits':['Successful full decode detects truncation/decoder errors, not every possible silent pixel modification.','Patient/study IDs and sections were checked against local source files; source release provenance and account authorization are not independently established by this audit.','Clinical correspondence of images and reports still requires the prescribed five-pair human review.','Processor/token truncation, 16-pair model execution, feature caching, and hypotheses.md are separate Step0 implementation requirements.']}
    (out/'audit_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    return 0 if all(c['passed'] for c in checks) else 1

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    raise SystemExit(main())
