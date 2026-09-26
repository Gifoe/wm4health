"""Secondary, frozen rollout-latent probe of device-computed future TCI reference CE."""
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,Subset
from core import *

OUT=ROOT/'outputs'
STEPS=[3,6,18,30]

def selected_train_indices(ds,store,per_patient=40):
    by_patient={}
    for i,(cid,t) in enumerate(ds.indices):
        subject=int(store.cases[int(cid)]['subjectid'])
        by_patient.setdefault(subject,[]).append(i)
    selected=[]
    for ids in by_patient.values():
        chosen=np.unique(np.linspace(0,len(ids)-1,min(per_patient,len(ids))).astype(int))
        selected.extend(ids[q] for q in chosen)
    return np.array(selected,np.int32)

def fit_ridge(x,y,subjects):
    good=np.isfinite(y);x=x[good].astype(np.float64);y=y[good].astype(np.float64);s=subjects[good]
    unique,count=np.unique(s,return_counts=True);n=dict(zip(unique,count))
    w=np.array([1/n[v] for v in s]);w*=len(w)/w.sum()
    mean=np.average(x,axis=0,weights=w);centered=x-mean
    std=np.sqrt(np.average(centered**2,axis=0,weights=w));std[std<1e-6]=1
    x=centered/std;intercept=np.average(y,weights=w)
    gram=x.T@(x*w[:,None]);rhs=x.T@((y-intercept)*w)
    coef=np.linalg.solve(gram+CFG['ridge_alpha']*np.eye(x.shape[1]),rhs)
    return {'coef':coef,'mean':mean,'std':std,'intercept':intercept}

def weighted_scores(y,p,subject):
    good=np.isfinite(y)&np.isfinite(p);y=y[good];p=p[good];s=subject[good]
    if not len(y):return np.nan,np.nan,np.nan,0,0
    unique,count=np.unique(s,return_counts=True);lookup=dict(zip(unique,count))
    w=np.array([1/lookup[k] for k in s]);w/=w.sum()
    ym=np.sum(w*y);pm=np.sum(w*p)
    var=np.sum(w*(y-ym)**2);pvar=np.sum(w*(p-pm)**2)
    r2=1-np.sum(w*(y-p)**2)/max(var,1e-12)
    corr=np.sum(w*(y-ym)*(p-pm))/max(np.sqrt(var*pvar),1e-12)
    mae=np.sum(w*np.abs(y-p))
    return float(r2),float(mae),float(corr),len(y),len(unique)

@torch.no_grad()
def latent_stream(model,ds,store,donors=None,policy='true'):
    base=ds.dataset if isinstance(ds,Subset) else ds
    for b in DataLoader(ds,batch_size=256,num_workers=0,pin_memory=True):
        idx=b['index'].numpy()
        if donors is not None:
            f=b['future_action'].numpy()
            for q,i in enumerate(idx):
                donor=donors[i]
                if donor>=0:
                    cid,t=base.indices[donor];c=store.cases[int(cid)]
                    f[q]=np.concatenate([c['an'][t+1:t+F+1],c['am'][t+1:t+F+1]],-1)
            b['future_action']=torch.from_numpy(f)
        b=to_device(b,'cuda')
        _,_,z=model(b,policy,return_steps=True)
        ce=np.stack([store.cases[int(base.indices[i,0])]['ce'][base.indices[i,1]+1:base.indices[i,1]+F+1]
                     for i in idx])
        subjects=np.array([int(store.cases[int(base.indices[i,0])]['subjectid']) for i in idx])
        yield idx,z.cpu().numpy(),ce,subjects

def main():
    seed_all();store=Store();train=Windows(store,'train');test=Windows(store,'test')
    ref=np.load(OUT/'test_reference.npz');donors=ref['donor']
    model=RSSM().cuda();model.set_action_stats(store)
    ck=torch.load(OUT/'checkpoints/true.pt',map_location='cuda',weights_only=False)
    model.load_state_dict(ck['model']);model.eval()
    for p in model.parameters():p.requires_grad_(False)
    ids=selected_train_indices(train,store)
    chosen=Subset(train,ids.tolist())
    # Subset uses base Dataset indices; latent_stream expects the same base index map.
    xs=[];ys=[];ss=[]
    for _,z,ce,subject in latent_stream(model,chosen,store):
        xs.append(z.reshape(-1,z.shape[-1]));ys.append(ce.reshape(-1,2));ss.append(np.repeat(subject,F))
    x=np.concatenate(xs);y=np.concatenate(ys);subject=np.concatenate(ss)
    probes=[fit_ridge(x,y[:,j],subject) for j in range(2)]
    for j,drug in enumerate(['PPF_CE','RFTN_CE']):np.savez(OUT/f'rollout_probe_{drug}.npz',**probes[j])
    rows=[];plot_data={}
    for condition,donor in [('true',None),('hold',None),('wrong',donors)]:
        # Hold: provide hold action as prospective schedule to the trained model.
        pred=[];refs=[];subjects=[];indices=[]
        for idx,z,ce,s in latent_stream(model,test,store,donors=donor,policy='hold' if condition=='hold' else 'true'):
            pp=np.stack([((z-p['mean'])/p['std'])@p['coef']+p['intercept'] for p in probes],-1)
            pred.append(pp);refs.append(ce);subjects.append(s);indices.append(idx)
        p=np.concatenate(pred);y=np.concatenate(refs);s=np.concatenate(subjects);index=np.concatenate(indices)
        assert np.array_equal(index,np.arange(len(test)))
        eligible=(donors>=0) if condition=='wrong' else np.ones(len(test),bool)
        for step in STEPS+[0]:
            take=range(F) if step==0 else [step-1]
            for j,drug in enumerate(['PPF_CE','RFTN_CE']):
                yy=y[eligible][:,take,j].reshape(-1);pp=p[eligible][:,take,j].reshape(-1)
                ss=np.repeat(s[eligible],len(take))
                r2,mae,corr,n,npat=weighted_scores(yy,pp,ss)
                rows.append({'condition':condition,'drug':drug,'horizon_seconds':step*10,
                             'r2':r2,'mae':mae,'pearson':corr,'n_points':n,'n_patients':npat})
        # Save only small deterministic illustration subset, not all rollout latents.
        plot_data[condition]=p[::max(1,len(p)//100),:,:]
        print('CE_PROBE',condition,flush=True)
    pd.DataFrame(rows).to_csv(OUT/'rollout_ce_probe_metrics.csv',index=False)
    np.savez_compressed(OUT/'rollout_ce_example.npz',**plot_data)
    (OUT/'rollout_ce_probe_summary.json').write_text(json.dumps({'train_anchor_samples':len(ids),
        'train_latent_step_samples':len(x),'probe_training_cases':'train only',
        'forecast_model_frozen':True,'ce_role':'device-computed TCI reference, not measured exposure'},indent=2))
    print('EXPOSURE_PROBE_COMPLETE',flush=True)

if __name__=='__main__':main()
