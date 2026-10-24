import json
import numpy as np
import pandas as pd
from .common import LABELS, Runtime, device_for, load_cache, load_config, manifest_path, output, parser, seed_all, write_json
from .metric import load_head, ranks, transform_cache


def group_masks(c, cache):
    positive=(cache['labels']==1)&(cache['mask']==1)
    groups=json.loads(output(c,'indices/frequency_groups.json').read_text(encoding='utf-8'))
    result={'all':np.ones(len(positive),dtype=bool)}
    for name,labels in groups.items():
        result[name]=positive[:,[LABELS.index(x) for x in labels]].any(1)
    result.update({name:positive[:,i] for i,name in enumerate(LABELS)})
    return result


def evaluate_one(c, cache, method, seed, split, prompts, device):
    head=load_head(c,method,seed,device)
    destination=output(c,'evaluation',split,method,f'seed_{seed}' if seed is not None else 'frozen',f'prompts_{prompts}')
    destination.mkdir(parents=True,exist_ok=True)
    with Runtime(device) as runtime:
        v,t=transform_cache(cache,head,device,prompts)
        scores=v@t.T
        rows=[];queries=[]
        masks=group_masks(c,cache)
        for direction,matrix in [('I2T',scores),('T2I',scores.T)]:
            rank=ranks(matrix);top=matrix.argmax(1)
            for group,keep in masks.items():
                r=rank[keep];n=len(r)
                rows.append({'method':method,'seed':seed,'split':split,'prompts':prompts,'direction':direction,'group':group,'queries':n,
                             'R@1':float((r<=1).mean()) if n else None,'R@5':float((r<=5).mean()) if n else None,
                             'R@10':float((r<=10).mean()) if n else None,'mAP':float((1/r).mean()) if n else None})
            for i in range(len(rank)):
                queries.append({'direction':direction,'query_id':int(cache['study_id'][i]),'candidate_id':int(cache['study_id'][top[i]]),
                                'correct_id':int(cache['study_id'][i]),'rank':int(rank[i]),'AP':float(1/rank[i]),
                                'correct_similarity':float(matrix[i,i]),'top1_similarity':float(matrix[i,top[i]]),
                                'high_frequency':bool(masks['high_frequency'][i]),'low_frequency':bool(masks['low_frequency'][i])})
    pd.DataFrame(rows).to_csv(destination/'metrics.csv',index=False)
    query=pd.DataFrame(queries);query.to_csv(destination/'query_ranks.csv',index=False)
    np.savez_compressed(destination/'features.npz',study_id=cache['study_id'],patient_id=cache['patient_id'],image_feature=v,text_feature=t,
                        labels=cache['labels'],mask=cache['mask'],branch=np.array('shared' if method in ['C','D'] else 'base' if method=='R0' else 'ordinary'))
    pd.DataFrame({'candidate_id':cache['study_id']}).to_csv(destination/'candidate_order.csv',index=False)
    frame=pd.read_csv(manifest_path(c)).set_index('study_id')
    examples=[]
    image_queries=query.loc[query.direction.eq('I2T')]
    for kind,part in [('success',image_queries.loc[image_queries['rank'].eq(1)].head(5)),('failure',image_queries.loc[image_queries['rank'].gt(1)].head(5))]:
        for row in part.to_dict('records'):
            row.update(kind=kind,correct_report=frame.loc[row['correct_id'],'report'],top1_report=frame.loc[row['candidate_id'],'report'])
            examples.append(row)
    pd.DataFrame(examples).to_csv(destination/'cases.csv',index=False)
    write_json(destination/'runtime.json',{'method':method,'seed':seed,'split':split,'prompts':prompts,**runtime.record()})
    return rows


def bootstrap_differences(c, split, methods, seeds):
    """Paired resampling of queries, never candidate libraries, matched by seed."""
    iterations=c['evaluation']['bootstrap_iterations'];rng=np.random.default_rng(c['seed']);rows=[]
    def query(method,seed,prompts):
        folder=f'seed_{seed}' if method!='R0' else 'frozen'
        return pd.read_csv(output(c,'evaluation',split,method,folder,f'prompts_{prompts}/query_ranks.csv'))
    comparisons=[('B','A'),('D','C'),('C','A'),('D','B')]
    for after,before in comparisons:
        if after not in methods or before not in methods:continue
        for seed in seeds:
            a=query(after,seed,1);b=query(before,seed,1)
            if not a[['direction','query_id']].equals(b[['direction','query_id']]):raise ValueError('Bootstrap query order differs')
            n=len(a)//2
            delta=((a['rank'].to_numpy().reshape(2,n)<=5).astype(float)-(b['rank'].to_numpy().reshape(2,n)<=5).astype(float)).mean(0)
            draw=rng.integers(0,n,size=(iterations,n));values=delta[draw].mean(1)
            lo,hi=np.quantile(values,[.025,.975])
            rows.append({'comparison':f'{after}-{before}','seed':seed,'metric':'bidirectional_R@5','delta':float(delta.mean()),
                         'lower_95':float(lo),'upper_95':float(hi),'iterations':iterations,'resampling':'paired queries; fixed candidate library'})
    return rows


def main():
    p=parser('evaluate_retrieval: fixed-library bidirectional R@1/5/10 and mAP')
    p.add_argument('--method',choices=['ALL','R0','A','B','C','D'],default='ALL')
    p.add_argument('--seed',type=int,choices=[42,43,44])
    p.add_argument('--split',choices=['val','test'])
    p.add_argument('--prompts',choices=['1','5','both'],default='both')
    p.add_argument('--device',choices=['cuda','cpu'])
    args=p.parse_args();c=load_config(args.config);seed_all(c['seed']);device=device_for(c,args.device)
    split=args.split or c['evaluation']['split'];cache=load_cache(c,split)
    methods=c['methods'] if args.method=='ALL' else [args.method]
    seeds=[args.seed] if args.seed is not None else c['training']['seeds']
    prompt_counts=c['evaluation']['prompt_counts'] if args.prompts=='both' else [int(args.prompts)]
    rows=[]
    with Runtime(device) as runtime:
        for method in methods:
            for seed in ([None] if method=='R0' else seeds):
                for prompts in prompt_counts:
                    rows.extend(evaluate_one(c,cache,method,seed,split,prompts,device))
                    print(f'Evaluated {split} {method} seed={seed} prompts={prompts}',flush=True)
    result=pd.DataFrame(rows);base=output(c,'evaluation',split)
    result.to_csv(base/'metrics.csv',index=False)
    result.groupby(['method','prompts','direction','group'])[['R@1','R@5','R@10','mAP']].agg(['mean','std']).to_csv(base/'mean_std.csv')
    if 1 in prompt_counts:
        write_json(base/'paired_bootstrap.json',bootstrap_differences(c,split,methods,seeds))
    record={'methods':methods,'training_seeds':seeds,'split':split,**runtime.record()}
    write_json(base/'runtime.json',record)
    print('PASS: retrieval results in',base,flush=True)
