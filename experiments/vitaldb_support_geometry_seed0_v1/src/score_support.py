"""TRAIN-fitted latent/action/raw support distances; no future outcomes are read."""
import json,time
import faiss
import numpy as np
import pandas as pd
import torch
from common import *

faiss.omp_set_num_threads(8)

def arr(split,name):return np.load(ARRAYS/f'{split}_{name}.npy',mmap_mode='r')

def summary(a):
    # Input is Round-2 standardized log(1+10*dose); all summaries are schedule-only.
    d=np.diff(a,axis=1)
    return np.concatenate([a.mean(1),a.max(1),a[:,-1],a.sum(1)/30,
                           np.abs(d).mean(1),d.mean(1),np.abs(d).max(1)],1).astype('float32')

def fit_scale(x,robust=False):
    x=np.asarray(x)
    if robust:
        mean=np.median(x,axis=0);scale=(np.quantile(x,.75,axis=0)-np.quantile(x,.25,axis=0))/1.349
        # Tied infusion summaries require a nonzero fallback.
        scale=np.where(scale>1e-4,scale,np.std(x,axis=0))
    else:mean=x.mean(0);scale=x.std(0)
    return mean.astype('float32'),np.maximum(scale,1e-3).astype('float32')

def scaled(x,stats,block=True):
    y=np.clip((np.asarray(x)-stats[0])/stats[1],-8,8).astype('float32')
    if block:y/=np.sqrt(y.shape[1])
    return np.ascontiguousarray(y)

def index_for(x,key):
    dim=x.shape[1];nlist=256 if len(x)>50000 else 64
    idx=faiss.IndexIVFFlat(faiss.IndexFlatL2(dim),dim,nlist,faiss.METRIC_L2)
    rng=np.random.default_rng(73)
    sample=np.asarray(x[rng.choice(len(x),min(len(x),50000),replace=False)],dtype='float32')
    idx.train(sample);idx.add(np.asarray(x,dtype='float32'));idx.nprobe=24
    print('index',key,len(x),dim,flush=True)
    return idx

def search(idx,x,k=20):
    outd=[];outi=[]
    for start in range(0,len(x),4096):
        d,i=idx.search(np.ascontiguousarray(x[start:start+4096]),k)
        outd.append(d);outi.append(i)
    return np.concatenate(outd),np.concatenate(outi)

def maha(x,mean,cov):
    cov=np.asarray(cov,dtype='float64')
    ridge=.05*np.trace(cov)/len(cov)
    cov=(1-.05)*cov+ridge*np.eye(len(cov))
    chol=np.linalg.cholesky(cov)
    return np.sqrt(np.sum(np.linalg.solve(chol,(x-mean).T)**2,axis=0)).astype('float32'),chol

def local_pca(q,bank,neighbors):
    """Residual to 90%-variance local tangent; small neighbor Gram matrix on GPU."""
    device='cuda' if torch.cuda.is_available() else 'cpu'
    residual=np.empty(len(q),np.float32);dims=np.empty(len(q),np.int16)
    radius=np.empty(len(q),np.float32)
    for start in range(0,len(q),256):
        stop=min(start+256,len(q));v=torch.as_tensor(bank[neighbors[start:stop]],device=device)
        center=v.mean(1,keepdim=True);x=v-center
        delta=torch.as_tensor(q[start:stop],device=device)-center[:,0]
        gram=x@x.transpose(1,2)
        eig,vec=torch.linalg.eigh(gram)
        eig=eig.flip(1).clamp_min(0);vec=vec.flip(2)
        n=(eig.cumsum(1)<(.90*eig.sum(1,keepdim=True))).sum(1)+1
        n=torch.minimum(n,torch.full_like(n,x.shape[1]-1))
        proj=(x@delta[:,:,None]).squeeze(-1)
        coeff=(vec.transpose(1,2)@proj[:,:,None]).squeeze(-1)
        k=torch.arange(eig.shape[1],device=device)[None,:]<n[:,None]
        explained=torch.where(k,coeff.square()/eig.clamp_min(1e-8),0).sum(1)
        orth=(delta.square().sum(1)-explained).clamp_min(0).sqrt()
        residual[start:stop]=orth.cpu().numpy();dims[start:stop]=n.cpu().numpy()
        radius[start:stop]=x.square().sum(2).mean(1).sqrt().cpu().numpy()
        if start%20480==0:print('pca',start,len(q),flush=True)
    return residual,dims,radius

def rarity_fit(a):
    # Dose marginal only: 2 drug-specific TRAIN histograms, frozen before VAL/TEST.
    flat=np.asarray(a).reshape(-1,2);result=[]
    for j in range(2):
        hi=float(np.quantile(flat[:,j],.999));lo=float(np.min(flat[:,j]));edges=np.linspace(lo,hi,41)
        bins=np.clip(np.searchsorted(edges,flat[:,j],side='right')-1,0,40)
        counts=np.bincount(bins,minlength=41).astype(float)+1
        result.append((edges.astype('float32'),(counts/counts.sum()).astype('float32')))
    return result

def rarity(a,protocol):
    scores=[]
    for j,(edges,p) in enumerate(protocol):
        bins=np.clip(np.searchsorted(edges,a[:,:,j],side='right')-1,0,40)
        scores.append((-np.log(p[bins])).mean(1))
    return np.mean(scores,axis=0).astype('float32')

def main():
    tr_z=np.asarray(arr('train','z'));tr_a=np.asarray(arr('train','action'))
    tr_sum=summary(tr_a);tr_flat=tr_a.reshape(len(tr_a),-1)
    tr_raw=np.concatenate([np.asarray(arr('train','current')),np.asarray(arr('train','demo'))],1)
    state_stat=fit_scale(tr_z);flat_stat=fit_scale(tr_flat);sum_stat=fit_scale(tr_sum,robust=True)
    raw_stat=fit_scale(tr_raw)
    train_state=scaled(tr_z,state_stat);train_flat=scaled(tr_flat,flat_stat)
    train_sum=scaled(tr_sum,sum_stat);train_raw=scaled(tr_raw,raw_stat)
    train_joint=np.ascontiguousarray(np.concatenate([train_state,train_sum],1))
    indices={name:index_for(x,name) for name,x in [('state',train_state),('flat',train_flat),
                 ('summary',train_sum),('joint',train_joint),('raw',train_raw)]}
    rarity_protocol=rarity_fit(tr_a)
    state_cov=np.cov(train_state,rowvar=False);joint_cov=np.cov(train_joint,rowvar=False)
    state_mean=train_state.mean(0);joint_mean=train_joint.mean(0)
    # One fixed random latent replacement; unlike an orthogonal rotation it breaks
    # patient/action alignment and is a genuine representation negative control.
    rng=np.random.default_rng(31)
    random_bank=np.ascontiguousarray(rng.normal(0,1/np.sqrt(64),train_state.shape).astype('float32'))
    random_index=index_for(random_bank,'random_latent')
    random_joint_index=index_for(np.ascontiguousarray(np.concatenate([random_bank,train_sum],1)),'random_joint')
    geom={};all_scores={};ann_audit={}
    for split in ['val','test']:
        z=np.asarray(arr(split,'z'));a=np.asarray(arr(split,'action'))
        qstate=scaled(z,state_stat);qflat=scaled(a.reshape(len(a),-1),flat_stat)
        qsum=scaled(summary(a),sum_stat);qjoint=np.ascontiguousarray(np.concatenate([qstate,qsum],1))
        qraw=scaled(np.concatenate([np.asarray(arr(split,'current')),np.asarray(arr(split,'demo'))],1),raw_stat)
        ds,ns=search(indices['state'],qstate,100)
        df,_=search(indices['flat'],qflat,20)
        dsum,_=search(indices['summary'],qsum,20)
        dj,nj=search(indices['joint'],qjoint,20)
        if split=='val':
            # Exact retrieval on a fixed small validation sample audits IVF recall.
            check=np.random.default_rng(11).choice(len(z),size=128,replace=False)
            for key,bank,query in [('state',train_state,qstate),('flat',train_flat,qflat),
                                   ('joint',train_joint,qjoint)]:
                exact=faiss.IndexFlatL2(bank.shape[1]);exact.add(bank)
                de,ie=exact.search(query[check],20)
                da,ia=indices[key].search(query[check],20)
                ann_audit[key]={'recall_at_20':float(np.mean([len(set(e)&set(a))/20 for e,a in zip(ie,ia)])),
                                'median_relative_20th_distance_error':float(np.median((np.sqrt(da[:,19])-np.sqrt(de[:,19]))/
                                                                                 np.maximum(np.sqrt(de[:,19]),1e-6)))}
                del exact
        dr,_=search(indices['raw'],qraw,20)
        sm,_=maha(qstate,state_mean,state_cov);jm,_=maha(qjoint,joint_mean,joint_cov)
        cond=np.empty((len(z),3),np.float32)
        for st in range(0,len(z),1024):
            en=min(st+1024,len(z));local=train_flat[ns[st:en]]
            delta=np.linalg.norm(local-qflat[st:en,None,:],axis=2)
            for j,k in enumerate(CFG['conditional_k']):cond[st:en,j]=delta[:,:k].min(1)
        perp,dim,radius=local_pca(qjoint,train_joint,nj)
        random_query=np.ascontiguousarray(rng.normal(0,1/np.sqrt(64),qstate.shape).astype('float32'))
        rand,_=search(random_index,random_query,20)
        rand_joint,_=search(random_joint_index,np.ascontiguousarray(np.concatenate([random_query,qsum],1)),20)
        scores=pd.DataFrame({'D_state_knn':np.sqrt(ds[:,19]),'D_state_mahalanobis':sm,
          'D_action':np.sqrt(df[:,19]),'D_action_summary':np.sqrt(dsum[:,19]),
          'D_joint_knn':np.sqrt(dj[:,19]),'D_joint_mahalanobis':jm,
          'D_cond_20':cond[:,0],'D_cond_50':cond[:,1],'D_cond_100':cond[:,2],
          'D_perp':perp,'D_raw_physiology':np.sqrt(dr[:,19]),
          'D_action_rarity':rarity(a,rarity_protocol),'D_random_latent':np.sqrt(rand[:,19]),
          'D_random_joint':np.sqrt(rand_joint[:,19]),
          'tangent_dimension':dim,'local_radius':radius})
        scores.to_csv(OUT/f'{split}_support_scores.csv',index=False)
        geom[split]={'n':len(z),'tangent_dimension_median':float(np.median(dim)),
                     'tangent_dimension_q10_q90':np.quantile(dim,[.1,.9]).tolist(),
                     'local_radius_median':float(np.median(radius))}
        all_scores[split]=scores
        print('SCORED',split,len(scores),flush=True)
    # Keep selection on validation only. Validation ensemble errors are computed
    # without test inspection; choose conditional k by top-20% average precision.
    from sklearn.metrics import average_precision_score
    target=np.asarray(arr('val','target'))[:,:,0];mask=np.asarray(arr('val','mask'))
    preds=np.stack([np.load(ARRAYS/f'val_pred_seed{s}.npy',mmap_mode='r')[:,:,0] for s in CFG['seeds']])
    err=np.nanmean(np.where(mask,np.abs(preds.mean(0)-target),np.nan),1)
    valid=np.isfinite(err);label=err[valid]>=np.quantile(err[valid],.8)
    subject=np.load(ARRAYS/'val_subject.npy')[valid]
    _,inverse,count=np.unique(subject,return_inverse=True,return_counts=True)
    patient_weight=1/count[inverse]
    ap={k:float(average_precision_score(label,all_scores['val'][f'D_cond_{k}'][valid],
                                      sample_weight=patient_weight))
        for k in CFG['conditional_k']}
    chosen=max(CFG['conditional_k'],key=lambda k:ap[k])
    test=all_scores['test'].copy();test['D_cond']=test[f'D_cond_{chosen}']
    idx=np.asarray(arr('test','index'))
    test.insert(0,'anchor_t',idx[:,1]);test.insert(0,'caseid',idx[:,0])
    test.to_csv(OUT/'support_scores.csv',index=False)
    jwrite('support_reference_summary.json',{'train_bank_windows':len(tr_z),'state_dim':64,'action_flat_dim':60,
      'action_summary_dim':14,'joint_dim':78,'knn_k':20,'faiss_index':'IVFFlat nlist=256 nprobe=24',
      'distance':'Euclidean after TRAIN fitted clipping/normalization; kNN is 20th neighbor distance',
      'joint_block_normalization':'state and summary each divided by sqrt(block dimension)',
      'mahalanobis_shrinkage':.05,'conditional_k_validation_ap':ap,'conditional_k_chosen':chosen,
      'approximate_index_validation_audit':ann_audit,
      'local_pca_variance':.90,'local_pca_neighbors':20,'diagnostics':geom,
      'random_control':'independent Gaussian replacement of 64-d latent, fixed seed31; rotation omitted because Euclidean is invariant',
      'no_test_future_outcome_or_ce_used':True})
    print('SUPPORT_COMPLETE',flush=True)

if __name__=='__main__':main()
