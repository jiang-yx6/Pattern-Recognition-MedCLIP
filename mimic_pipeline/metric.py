import json
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from .common import Runtime, device_for, load_cache, load_config, output, parser, resolve, seed_all, sha256, write_json


class MetricHead(nn.Module):
    def __init__(self, dimension, hidden, method):
        super().__init__()
        if method not in ['A','B','C','D']:
            raise ValueError('Metric heads are defined for A/B/C/D')
        self.method = method
        self.image_shared = nn.Linear(dimension, hidden)
        self.text_shared = nn.Linear(dimension, hidden)
        if method in ['C','D']:
            self.image_private = nn.Linear(dimension, hidden)
            self.text_private = nn.Linear(dimension, hidden)
            self.image_reconstruct = nn.Linear(2*hidden, dimension)
            self.text_reconstruct = nn.Linear(2*hidden, dimension)

    def forward(self, v, t):
        r = {'image_feature':F.normalize(self.image_shared(v),dim=-1),
             'text_feature':F.normalize(self.text_shared(t),dim=-1)}
        if self.method in ['C','D']:
            r['image_private'] = F.normalize(self.image_private(v),dim=-1)
            r['text_private'] = F.normalize(self.text_private(t),dim=-1)
            r['image_reconstruction'] = self.image_reconstruct(torch.cat([r['image_feature'],r['image_private']],dim=1))
            r['text_reconstruction'] = self.text_reconstruct(torch.cat([r['text_feature'],r['text_private']],dim=1))
        return r


def valid_negatives(labels, masks, patient_ids):
    positive = (labels == 1) & (masks == 1)
    identical = (positive[:,None,:] == positive[None,:,:]).all(-1)
    nonempty = positive.any(-1)
    same_patient = patient_ids[:,None] == patient_ids[None,:]
    return ~(same_patient | (identical & nonempty[:,None] & nonempty[None,:]))


def negative_weights(similarity, valid, hard_fraction, hard_weight):
    """Rank only valid negatives, ceil(25%); stable ties use candidate position."""
    count = valid.sum(1)
    k = torch.ceil(count.float()*hard_fraction).long()
    ordered = torch.argsort(similarity.masked_fill(~valid,-torch.inf),dim=1,descending=True,stable=True)
    hard_sorted = torch.arange(similarity.shape[1],device=similarity.device)[None,:] < k[:,None]
    hard = torch.zeros_like(valid).scatter(1,ordered,hard_sorted) & valid
    weights = torch.ones_like(similarity)
    weights[hard] = hard_weight
    return weights, hard


def contrastive_loss(v, t, labels, masks, patient_ids, temperature=0.07, hard=False, fraction=0.25, weight=2.):
    similarity = v @ t.T
    valid = valid_negatives(labels,masks,patient_ids)
    eye = torch.eye(len(v),dtype=torch.bool,device=v.device)
    losses = []; selected = []; skipped = 0
    for scores, candidates in [(similarity,valid),(similarity.T,valid.T)]:
        active = candidates.any(1); skipped += int((~active).sum().item())
        if hard:
            weights, chosen = negative_weights(scores,candidates,fraction,weight)
        else:
            weights = torch.ones_like(scores); chosen = torch.zeros_like(candidates)
        logits = (scores/temperature + weights.log()).masked_fill(~(candidates|eye),-torch.inf)
        if active.any():
            losses.append((torch.logsumexp(logits,dim=1)-scores.diagonal()/temperature)[active].mean())
        selected.append(chosen)
    return (torch.stack(losses).mean() if losses else None), skipped, selected


def auxiliary_losses(encoded, v, t):
    if 'image_private' not in encoded:
        zero = v.new_zeros(())
        return zero,zero
    reconstruction = (F.mse_loss(encoded['image_reconstruction'],v)+F.mse_loss(encoded['text_reconstruction'],t))/2
    covariances = []
    for modality in ['image','text']:
        s = encoded[modality+'_feature']; p = encoded[modality+'_private']
        cross = (s-s.mean(0)).T @ (p-p.mean(0)) / max(len(s)-1,1)
        covariances.append(cross.square().mean())
    return reconstruction,torch.stack(covariances).mean()


def ranks(scores):
    """One true item on the diagonal; ties resolved by fixed candidate order."""
    scores = np.asarray(scores)
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1] or not np.isfinite(scores).all():
        raise ValueError('Retrieval needs a finite square same-study similarity matrix')
    n = len(scores); truth = np.diag(scores)[:,None]
    ties = np.arange(n)[None,:] < np.arange(n)[:,None]
    return 1 + ((scores > truth) | ((scores == truth) & ties)).sum(1)


def retrieval_values(scores):
    result = {}
    for direction,x in [('I2T',scores),('T2I',scores.T)]:
        r = ranks(x)
        result[direction] = {'R@1':float((r<=1).mean()),'R@5':float((r<=5).mean()),
                             'R@10':float((r<=10).mean()),'mAP':float((1/r).mean())}
    return result


def run_dir(c, method, seed, smoke=False):
    return output(c,'smoke_training' if smoke else 'training',method,f'seed_{seed}')


def load_head(c, method, seed, device):
    if method == 'R0':
        return None
    checkpoint = run_dir(c,method,seed)/'model.pt'
    state = torch.load(checkpoint,map_location='cpu',weights_only=True)
    if state['method'] != method or state['seed'] != seed or state['manifest_sha256'] != c['expected_manifest_sha256']:
        raise ValueError('Metric checkpoint provenance mismatch')
    if state['train_cache_sha256'] != sha256(output(c,'features/train.npz')):
        raise ValueError('Training cache differs from the metric checkpoint')
    head = MetricHead(c['feature_dimension'],c['training']['head_dimension'],method)
    head.load_state_dict(state['state_dict'],strict=True)
    return head.eval().to(device)


def transform_cache(cache, head, device, prompts=1, branch='shared'):
    if prompts not in [1,5]:
        raise ValueError('Use one or five prompt variants')
    variants = cache['text_prompt_features']
    if variants.shape[0] < prompts:
        raise ValueError('Missing prompt variants: rerun extract_features for this split')
    with torch.inference_mode():
        v = torch.as_tensor(cache['image_feature'],device=device)
        t = torch.as_tensor(variants[:prompts],device=device)
        if head is None:
            visual = v; texts = t
        else:
            v_head = head.image_private if branch=='private' else head.image_shared
            t_head = head.text_private if branch=='private' else head.text_shared
            visual = F.normalize(v_head(v),dim=-1)
            texts = F.normalize(t_head(t),dim=-1)
        # Apply the method's head to EACH version, normalize, mean, normalize again.
        textual = F.normalize(texts.mean(0),dim=-1)
    return visual.cpu().numpy(),textual.cpu().numpy()


def train_one(c, method, seed, train, val, device, smoke=False, force=False):
    directory = run_dir(c,method,seed,smoke);directory.mkdir(parents=True,exist_ok=True)
    cfg_sha = sha256(c['_config_path']); cache_sha = sha256(output(c,'features/train.npz'))
    log_path = directory/'runtime.json'
    if log_path.exists() and not force:
        old = json.loads(log_path.read_text(encoding='utf-8'))
        if old['config_sha256']==cfg_sha and old['train_cache_sha256']==cache_sha and (directory/'model.pt').exists():
            print(f'Reusing completed {method} seed {seed}',flush=True);return old
        raise ValueError('Existing run uses different inputs; pass --force explicitly')
    seed_all(seed)
    settings = c['training']; head = MetricHead(c['feature_dimension'],settings['head_dimension'],method).to(device)
    optimizer = torch.optim.AdamW(head.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
    n = min(128,len(train['study_id'])) if smoke else len(train['study_id'])
    tensors = {key:torch.as_tensor(train[key][:n],device=device) for key in ['image_feature','text_feature','labels','mask','patient_id']}
    vv = torch.as_tensor(val['image_feature'],device=device); vt = torch.as_tensor(val['text_feature'],device=device)
    rng = np.random.default_rng(seed)
    epochs = 2 if smoke else settings['max_epochs']
    best = -1.; stale = 0; history = []; initial = {k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
    gradients = {name:False for name,_ in head.named_parameters()}
    examples = []; total_skipped=0
    with Runtime(device) as runtime:
        for epoch in range(1,epochs+1):
            head.train(); order = rng.permutation(n); records=[]; skipped=0
            for start in range(0,n,settings['batch_size']):
                ix = order[start:start+settings['batch_size']]
                if len(ix)<2:
                    skipped += 2*len(ix);continue
                v=tensors['image_feature'][ix];t=tensors['text_feature'][ix]
                encoded=head(v,t)
                loss, skip, hard_masks=contrastive_loss(encoded['image_feature'],encoded['text_feature'],tensors['labels'][ix],tensors['mask'][ix],tensors['patient_id'][ix],
                    settings['temperature'],method in ['B','D'],c['mining']['hard_fraction'],c['mining']['hard_weight'])
                skipped+=skip
                if loss is None:
                    continue
                reconstruction,covariance=auxiliary_losses(encoded,v,t)
                total=loss+settings['reconstruction_weight']*reconstruction+settings['covariance_weight']*covariance
                if not torch.isfinite(total):
                    raise ValueError('Training loss is nonfinite')
                optimizer.zero_grad(set_to_none=True);total.backward()
                for name,parameter in head.named_parameters():
                    if parameter.grad is not None:
                        if not torch.isfinite(parameter.grad).all():raise ValueError('Nonfinite gradient')
                        gradients[name] |= bool(parameter.grad.abs().sum().item()>0)
                optimizer.step()
                records.append([float(total.item()),float(loss.item()),float(reconstruction.item()),float(covariance.item())])
                if len(examples)<20 and method in ['B','D']:
                    similarity=(encoded['image_feature']@encoded['text_feature'].T).detach().cpu().numpy()
                    for row,col in hard_masks[0].nonzero().tolist():
                        q=int(ix[row]);k=int(ix[col])
                        positive=(train['labels']==1)&(train['mask']==1)
                        examples.append({'query_id':int(train['study_id'][q]),'candidate_id':int(train['study_id'][k]),'direction':'I2T',
                                         'similarity':float(similarity[row,col]),'weight':c['mining']['hard_weight'],
                                         'label_overlap':int((positive[q]&positive[k]).sum()),'epoch':epoch})
                        if len(examples)>=20:break
            if not records:raise ValueError('No query has valid negatives in this training run')
            head.eval()
            with torch.inference_mode():
                result=head(vv,vt);metrics=retrieval_values((result['image_feature']@result['text_feature'].T).cpu().numpy())
            score=(metrics['I2T']['R@5']+metrics['T2I']['R@5'])/2
            mean=np.mean(records,axis=0)
            history.append({'epoch':epoch,'total_loss':float(mean[0]),'contrastive_loss':float(mean[1]),'reconstruction_loss':float(mean[2]),
                            'covariance_loss':float(mean[3]),'val_R5':score,'skipped_queries':skipped})
            total_skipped+=skipped
            print(f'{method} seed={seed} epoch={epoch}: loss={mean[0]:.5f} val_R5={score:.4f} skipped={skipped}',flush=True)
            if score>best:
                best=score;stale=0;best_epoch=epoch
                torch.save({'state_dict':{k:v.detach().cpu().clone() for k,v in head.state_dict().items()},'method':method,'seed':seed,
                            'manifest_sha256':c['expected_manifest_sha256'],'train_cache_sha256':cache_sha,'config_sha256':cfg_sha,
                            'input_dimension':c['feature_dimension'],'head_dimension':settings['head_dimension'],'smoke_test':smoke},directory/'model.pt')
            else:stale+=1
            if not smoke and stale>=settings['patience']:break
    changed=any(not torch.equal(initial[k],v.detach().cpu()) for k,v in head.state_dict().items())
    if not changed or not all(gradients.values()):raise ValueError('Some heads failed to receive gradients or update')
    head.load_state_dict(torch.load(directory/'model.pt',map_location=device,weights_only=True)['state_dict'])
    head.eval();transformed=head(vv,vt)
    payload={key:val[key] for key in ['patient_id','study_id','labels','mask']}
    payload.update(image_feature=transformed['image_feature'].detach().cpu().numpy(),text_feature=transformed['text_feature'].detach().cpu().numpy(),
                   branch=np.array('shared' if method in ['C','D'] else 'ordinary'))
    if method in ['C','D']:
        payload.update(image_private=transformed['image_private'].detach().cpu().numpy(),text_private=transformed['text_private'].detach().cpu().numpy())
    np.savez_compressed(directory/'val_features.npz',**payload)
    pd.DataFrame(history).to_csv(directory/'losses.csv',index=False)
    if examples:pd.DataFrame(examples).to_csv(directory/'hard_examples_20.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,3))
    for ax,name in zip(axes,['contrastive_loss','reconstruction_loss','covariance_loss']):
        ax.plot([x['epoch'] for x in history],[x[name] for x in history])
        ax.set(xlabel='Epoch',ylabel=name)
    fig.tight_layout();fig.savefig(directory/'losses.png');plt.close(fig)
    std={key:np.std(value,axis=0).tolist() for key,value in payload.items() if key in ['image_feature','text_feature','image_private','text_private']}
    write_json(directory/'per_dimension_std.json',std)
    record={'method':method,'seed':seed,'smoke_test':smoke,'status':'complete','epochs':len(history),'best_epoch':best_epoch,'best_val_R5':best,
            'training_samples':n,'trainable_parameters':sum(x.numel() for x in head.parameters()),'head_parameters_updated':changed,
            'gradient_checks':gradients,'backbone_loaded':False,'backbone_updated':False,'backbone_frozen':True,'skipped_queries':total_skipped,
            'checkpoint_sha256':c['expected_checkpoint_sha256'],'train_cache_sha256':cache_sha,'config_sha256':cfg_sha,**runtime.record()}
    write_json(log_path,record)
    return record


def main():
    p=parser('train_metric: cached-feature A/B/C/D training with seeds 42,43,44')
    p.add_argument('--method',choices=['ALL','A','B','C','D'],default='ALL')
    p.add_argument('--seed',type=int,choices=[42,43,44])
    p.add_argument('--device',choices=['cuda','cpu'])
    p.add_argument('--smoke-test',action='store_true',help='128 train pairs, two epochs, separate output directory')
    p.add_argument('--force',action='store_true')
    args=p.parse_args();c=load_config(args.config);device=device_for(c,args.device)
    train=load_cache(c,'train');val=load_cache(c,'val')
    methods=[m for m in c['methods'] if m!='R0'] if args.method=='ALL' else [args.method]
    seeds=[args.seed] if args.seed is not None else ([c['training']['seeds'][0]] if args.smoke_test else c['training']['seeds'])
    before=sha256(resolve(c['checkpoint']));records=[]
    for method in methods:
        for seed in seeds:records.append(train_one(c,method,seed,train,val,device,args.smoke_test,args.force))
    if sha256(resolve(c['checkpoint']))!=before:raise ValueError('Frozen backbone checkpoint changed')
    name='smoke_training' if args.smoke_test else 'training'
    write_json(output(c,name,'run_summary.json'),records)
    pd.DataFrame([{k:r[k] for k in ['method','seed','epochs','best_epoch','best_val_R5','elapsed_seconds','peak_cuda_memory_mib','trainable_parameters','smoke_test']} for r in records]).to_csv(output(c,'logs',f'{name}.csv'),index=False)
    print('PASS:',len(records),'training runs;',name,flush=True)
