import numpy as np
import pandas as pd
from .common import LABELS, Runtime, device_for, load_cache, load_config, output, parser, results, seed_all, write_json
from .metric import load_head, transform_cache


def space_statistics(v, t, labels, mask, seed=42, maximum_pairs=10000):
    if v.shape != t.shape or not np.isfinite(v).all() or not np.isfinite(t).all():
        raise ValueError('Geometry needs aligned finite feature matrices')
    n=len(v);i,j=np.triu_indices(n,k=1)
    rng=np.random.default_rng(seed)
    sampled=rng.choice(len(i),size=min(maximum_pairs,len(i)),replace=False) if len(i) else np.array([],dtype=int)
    positive=(labels==1)&(mask==1);nonempty=positive.any(1)
    between=~(positive[i]&positive[j]).any(1)&nonempty[i]&nonempty[j]
    result={'samples':n,'dimension':v.shape[1],'alignment_squared_l2':float(np.square(v-t).sum(1).mean()),
            'paired_cosine_distance':float((1-(v*t).sum(1)).mean()),'modality_gap':float(np.linalg.norm(v.mean(0)-t.mean(0))),
            'uniformity_pairs':len(sampled),'seed':seed,'class_distances':[]}
    decompositions={}
    for modality,x in [('image',v),('text',t)]:
        pair_distance=np.clip(1-(x[i]*x[j]).sum(1),0,2)
        result[modality+'_uniformity']=float(np.log(np.exp(-4*pair_distance[sampled]).mean())) if len(sampled) else None
        inter=float(pair_distance[between].mean()) if between.any() else None
        for k,name in enumerate(LABELS):
            within=positive[i,k]&positive[j,k]
            intra=float(pair_distance[within].mean()) if within.any() else None
            disease=positive[:,k]
            result['class_distances'].append({'modality':modality,'disease':name,'within_pairs':int(within.sum()),
                'between_pairs':int(between.sum()),'within_cosine_distance':intra,'between_cosine_distance':inter,
                'within_between_ratio':intra/inter if intra is not None and inter is not None and inter>0 else None,
                'positive_samples':int(disease.sum()),'comorbidity_fraction':float((positive[disease].sum(1)>1).mean()) if disease.any() else None})
        centered=x.astype('float64')-x.mean(0,dtype='float64')
        singular=np.linalg.svd(centered,compute_uv=False);power=singular**2;total=power.sum()
        cumulative=np.cumsum(power)/total if total>0 else np.zeros(len(power))
        result[modality+'_energy_top10']=float(cumulative[min(10,len(cumulative))-1])
        result[modality+'_energy_top50']=float(cumulative[min(50,len(cumulative))-1])
        result[modality+'_dimensions_90']=int(np.searchsorted(cumulative,.9)+1) if total>0 else 0
        result[modality+'_mean_dimension_std']=float(centered.std(0).mean())
        decompositions[modality]={'singular_values':singular,'cumulative_energy':cumulative,'dimension_std':centered.std(0)}
    return result,decompositions


def analyze_one(c, cache, method, seed, device, visualization=True):
    head=load_head(c,method,seed,device)
    base=results(c,'geometry',method,f'seed_{seed}' if seed is not None else 'frozen');base.mkdir(parents=True,exist_ok=True)
    summaries=[]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with Runtime(device) as runtime:
        for branch in (['shared','private'] if method in ['C','D'] else ['base' if method=='R0' else 'ordinary']):
            v,t=transform_cache(cache,head,device,1,'private' if branch=='private' else 'shared')
            result,decompositions=space_statistics(v,t,cache['labels'],cache['mask'],c['seed'],c['geometry']['uniformity_pairs'])
            result.update(method=method,training_seed=seed,branch=branch,split='val')
            directory=base/branch;directory.mkdir(exist_ok=True)
            pd.DataFrame(result.pop('class_distances')).to_csv(directory/'class_distances.csv',index=False)
            write_json(directory/'metrics.json',result);summaries.append(result)
            fig,axes=plt.subplots(1,2,figsize=(9,3))
            for modality,values in decompositions.items():
                pd.DataFrame({'singular_value':values['singular_values'],'cumulative_energy':values['cumulative_energy']}).to_csv(directory/f'{modality}_svd.csv',index=False)
                pd.DataFrame({'dimension_std':values['dimension_std']}).to_csv(directory/f'{modality}_dimension_std.csv',index=False)
                axes[0].plot(values['singular_values'],label=modality);axes[1].plot(values['cumulative_energy'],label=modality)
            axes[0].set(xlabel='Component',ylabel='Singular value');axes[1].set(xlabel='Component',ylabel='Cumulative squared energy')
            axes[0].legend();fig.suptitle(f'{method} {seed} {branch}; val n={len(v)} seed={c["seed"]}');fig.tight_layout();fig.savefig(directory/'singular_spectrum.png');plt.close(fig)
            if visualization:
                from sklearn.decomposition import PCA
                from sklearn.manifold import TSNE
                ix=np.random.default_rng(c['seed']).choice(len(v),size=min(c['geometry']['max_visualization_pairs'],len(v)),replace=False)
                joint=np.concatenate([v[ix],t[ix]])
                reduced=PCA(n_components=min(50,joint.shape[1],len(joint)-1),random_state=c['seed']).fit_transform(joint)
                embedded=TSNE(n_components=2,perplexity=min(30,(len(joint)-1)/3),learning_rate='auto',init='pca',
                              max_iter=c['geometry']['tsne_iterations'],random_state=c['seed']).fit_transform(reduced)
                positives=(cache['labels'][ix]==1)&(cache['mask'][ix]==1)
                classes=np.where(positives.any(1),positives.argmax(1),-1)
                fig,ax=plt.subplots(figsize=(7,5))
                for label in [-1,0,1,2,3,4]:
                    keep=classes==label;color=plt.get_cmap('tab10')(label+1)
                    for offset,marker,modality in [(0,'o','image'),(len(ix),'^','text')]:
                        xy=embedded[offset:offset+len(ix)][keep]
                        ax.scatter(xy[:,0],xy[:,1],s=12,alpha=.65,c=[color],marker=marker,label=f'{LABELS[label] if label>=0 else "unknown"}/{modality}')
                ax.legend(fontsize=6);ax.set_title(f'{method} {seed} {branch}: joint val t-SNE, {len(ix)} pairs, seed={c["seed"]}')
                fig.tight_layout();fig.savefig(directory/'joint_tsne.png');plt.close(fig)
                pd.DataFrame({'study_id':np.tile(cache['study_id'][ix],2),'modality':['image']*len(ix)+['text']*len(ix),
                              'x':embedded[:,0],'y':embedded[:,1],'first_known_positive_class':np.tile(classes,2)}).to_csv(directory/'joint_tsne.csv',index=False)
            print('Analyzed',method,seed,branch,flush=True)
    write_json(base/'runtime.json',{'method':method,'seed':seed,'split':'val',**runtime.record()})
    return summaries


def main():
    p=parser('analyze_geometry: fixed-val alignment, uniformity, class distances, modality gap, SVD and joint t-SNE')
    p.add_argument('--method',choices=['ALL','R0','A','B','C','D'],default='ALL')
    p.add_argument('--seed',type=int,choices=[42,43,44])
    p.add_argument('--device',choices=['cuda','cpu'])
    p.add_argument('--skip-visualization',action='store_true')
    args=p.parse_args();c=load_config(args.config);seed_all(c['seed']);device=device_for(c,args.device)
    if c['geometry']['split']!='val':raise ValueError('Geometry comparisons must use the fixed val set')
    cache=load_cache(c,'val');rows=[]
    methods=c['methods'] if args.method=='ALL' else [args.method]
    seeds=[args.seed] if args.seed is not None else c['training']['seeds']
    for method in methods:
        for seed in ([None] if method=='R0' else seeds):rows.extend(analyze_one(c,cache,method,seed,device,not args.skip_visualization))
    pd.DataFrame(rows).to_csv(results(c,'geometry/metrics.csv'),index=False)
    print('PASS: fixed-val geometry analysis complete',flush=True)
