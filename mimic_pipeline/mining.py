import csv
import json
import numpy as np
import pandas as pd
import torch
from .common import LABELS, Runtime, device_for, load_cache, load_config, manifest_path, output, parser, results, seed_all, write_json
from .metric import load_head, negative_weights, transform_cache, valid_negatives


def conflict_flags(c, cache):
    with np.load(output(c,'features/disease_prompts.npz')) as z:
        prompts=z['text_feature'];scale=float(z['logit_scale'])
    similarities=np.einsum('nd,kpd->nkp',cache['image_feature'],prompts)
    score=((similarities[:,:,0]+similarities[:,:,1])/2-similarities[:,:,2])*scale
    confidence=1/(1+np.exp(-np.clip(score,-80,80)))
    conflict=((cache['labels']==1)&(confidence<c['mining']['conflict_positive_threshold']) |
              (cache['labels']==0)&(confidence>c['mining']['conflict_negative_threshold'])) & (cache['mask']==1)
    return conflict.any(1),confidence


def mine_one(c, cache, method, seed, mode, device):
    head=load_head(c,method,seed,device)
    v,t=transform_cache(cache,head,device,1)
    n=len(v);positive=(cache['labels']==1)&(cache['mask']==1)
    patient=torch.as_tensor(cache['patient_id']);labels=torch.as_tensor(cache['labels']);mask=torch.as_tensor(cache['mask'])
    conflict,confidence=conflict_flags(c,cache)
    directory=results(c,'negatives',method,f'seed_{seed}',mode);directory.mkdir(parents=True,exist_ok=True)
    fields=['query_id','candidate_id','direction','similarity','weight','conflict','label_overlap','selection']
    examples=[];counts={};overlap={};similarities={};groups={}
    frequency=json.loads(output(c,'indices/frequency_groups.json').read_text(encoding='utf-8'))
    order=np.random.default_rng(seed).permutation(n)
    with Runtime(device) as runtime, (directory/'negatives.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for direction,left,right in [('I2T',v,t),('T2I',t,v)]:
            batches=[order[i:i+c['training']['batch_size']] for i in range(0,n,c['training']['batch_size'])] if mode=='batch' else [np.arange(n)]
            for batch in batches:
                similarity=left[batch]@right[batch].T
                valid=valid_negatives(labels[batch],mask[batch],patient[batch])
                if mode=='batch':
                    _,hard=negative_weights(torch.from_numpy(similarity),valid,c['mining']['hard_fraction'],c['mining']['hard_weight'])
                    selections=[('top25percent',hard.numpy())]
                else:
                    ordered=np.argsort(-np.where(valid.numpy(),similarity,-np.inf),axis=1,kind='stable')
                    choices={name:np.zeros_like(similarity,dtype=bool) for name in ['top20','conflict_first20']}
                    for i in range(len(batch)):
                        candidates=ordered[i][valid[i,ordered[i]].numpy()][:c['mining']['top_candidates']]
                        first=sorted(candidates.tolist(),key=lambda j:not conflict[batch[j]])
                        choices['top20'][i,candidates[:c['mining']['budget']]]=True
                        choices['conflict_first20'][i,first[:c['mining']['budget']]]=True
                    selections=list(choices.items())
                for selection,chosen in selections:
                    for i,j in np.argwhere(chosen):
                        q=int(batch[i]);k=int(batch[j]);same=int((positive[q]&positive[k]).sum())
                        row={'query_id':int(cache['study_id'][q]),'candidate_id':int(cache['study_id'][k]),'direction':direction,
                             'similarity':float(similarity[i,j]),'weight':c['mining']['hard_weight'],'conflict':bool(conflict[k]),
                             'label_overlap':same,'selection':selection}
                        writer.writerow(row)
                        key=direction+'/'+selection;counts[key]=counts.get(key,0)+1
                        overlap[key]=overlap.get(key,0)+int(same>0);similarities[key]=similarities.get(key,0)+float(similarity[i,j])
                        group=groups.setdefault(key,{name:0 for name in frequency})
                        for name,names in frequency.items():group[name]+=int(positive[k,[LABELS.index(x) for x in names]].any())
                        if len(examples)<20:examples.append(row)
    reports=pd.read_csv(manifest_path(c)).set_index('study_id')
    for row in examples:
        row['query_report']=reports.loc[row['query_id'],'report'];row['candidate_report']=reports.loc[row['candidate_id'],'report']
    pd.DataFrame(examples).to_csv(directory/'review_20.csv',index=False)
    pd.DataFrame({'study_id':cache['study_id'],'conflict':conflict,**{name:confidence[:,i] for i,name in enumerate(LABELS)}}).to_csv(directory/'conflicts.csv',index=False)
    summary={'method':method,'seed':seed,'source_split':'train','mode':mode,'rounding':'ceil of valid negative count * 0.25',
             'counts':counts,'known_positive_overlap_rate':{k:overlap[k]/n for k,n in counts.items()},
             'mean_similarity':{k:similarities[k]/n for k,n in counts.items()},'candidate_frequency_group_counts':groups,
             'note':'Known-label overlap is a proxy, not a verified false-negative rate; no validation/test candidates are used.',**runtime.record()}
    write_json(directory/'runtime.json',summary)
    print('Mined',method,seed,mode,sum(counts.values()),'negatives',flush=True)
    return summary


def main():
    p=parser('mine_negatives: training-only batch hard negatives or Top50 fixed-budget prompt selection')
    p.add_argument('--method',choices=['R0','B','D'],default='B')
    p.add_argument('--seed',type=int,choices=[42,43,44])
    p.add_argument('--mode',choices=['batch','budget'],default='batch')
    p.add_argument('--device',choices=['cuda','cpu'])
    args=p.parse_args();c=load_config(args.config);seed_all(c['seed']);device=device_for(c,args.device)
    cache=load_cache(c,'train')
    seeds=[args.seed] if args.seed is not None else ([c['seed']] if args.method=='R0' else c['training']['seeds'])
    for seed in seeds:
        mine_one(c,cache,args.method,seed,args.mode,device)
