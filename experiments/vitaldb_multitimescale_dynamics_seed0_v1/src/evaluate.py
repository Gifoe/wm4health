"""Patient-weighted forecasting, paired inference, event and action audits."""
import json,hashlib,sys,warnings
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch.utils.data import DataLoader
from models import *

OUT=ROOT/'outputs';ARRAYS=OUT/'arrays';ARRAYS.mkdir(parents=True,exist_ok=True)
NAME=['Direct-MH','Direct-MH-Capacity','MT-Dynamics']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def patient_boot(values,subject,valid,reps=1000):
    m=valid&np.isfinite(values);frame=pd.DataFrame({'subject':subject[m],'value':values[m]})
    q=frame.groupby('subject').value.mean().to_numpy()
    if len(q)<2:return (float(q.mean()) if len(q) else np.nan,np.nan,np.nan,len(q),int(m.sum()))
    rng=np.random.default_rng(0);draws=q[rng.integers(len(q),size=(reps,len(q)))].mean(1)
    return float(q.mean()),float(np.quantile(draws,.025)),float(np.quantile(draws,.975)),len(q),int(m.sum())
def group_masks(ref,store):
    up=ref['upcoming_label'];n=len(up);cap=np.asarray(json.loads((SOURCE/'outputs/pump_rate_tail_protocol.json').read_text())['0.99']['maximum_ml_per_10s'])
    cur=np.asarray([store.cases[int(c)]['action'][int(t)] for c,t in zip(ref['case'],ref['t'])])
    clean=(ref['future_action']<=cap).all((1,2))&(cur<=cap).all(1)
    result={'overall':np.ones(n,bool),'stable_action_Q1':ref['quartile']==1,'Q4':ref['quartile']==4,
            'upcoming_large_intervention':up>0,'high_rate_tail':~clean,'non_tail':clean}
    for k,s in [(1,'initiation'),(2,'increase'),(3,'decrease'),(4,'stop')]:result[s]=up==k
    return result

@torch.no_grad()
def infer(model,ds,store,device,condition='true',donors=None,donor_actions=None,latent=False):
    model.eval();n=len(ds);pred=np.empty((n,4,2),np.float32);lat=[];at=0
    for b in DataLoader(ds,batch_size=512,num_workers=0,pin_memory=True):
        ids=b['index'].numpy();count=len(ids)
        if condition=='hold':b['future_action']=b['action'][:,-1:].expand(-1,F,-1).clone()
        elif condition=='wrong':
            a=b['future_action'].numpy().copy();valid=donors[ids]>=0;a[valid]=donor_actions[donors[ids[valid]]];b['future_action']=torch.from_numpy(a)
        b=to_device(b,device)
        if isinstance(model,Direct):
            p=model(b,STEPS);q=None
        elif latent:p,q=model(b,STEPS,True)
        else:p=model(b,STEPS);q=None
        pred[at:at+count]=p.cpu().numpy()*store.norm['state_std'][[0,2]]+store.norm['state_mean'][[0,2]]
        if latent:lat.append({k:v.cpu().numpy() for k,v in q.items()})
        at+=count
    if latent:
        keys=lat[0];return pred,{k:np.concatenate([r[k] for r in lat]) for k in keys}
    return pred,None

def metrics(pred,ref,groups):
    rows=[];subject=ref['subject'];target=ref['target'];mask=ref['mask']
    for group,keep in groups.items():
        for name,p in pred.items():
            for j,var in enumerate(('BIS','MAP')):
                for k,h in enumerate(STEPS):
                    valid=keep&mask[:,h-1,j];e=np.abs(p[:,k,j]-target[:,h-1,j]);sq=(p[:,k,j]-target[:,h-1,j])**2
                    a,_,_,n,w=patient_boot(e,subject,valid);mse,_,_,_,_=patient_boot(sq,subject,valid)
                    rows.append(dict(model=name,subset=group,target=var,horizon_seconds=h*10,mae=a,rmse=np.sqrt(mse),patients=n,windows=w))
    return pd.DataFrame(rows)

def paired(pred,ref,groups):
    rows=[];sub=ref['subject'];target=ref['target'];mask=ref['mask']
    for group,keep in groups.items():
        for j,var in enumerate(('BIS','MAP')):
            for k,h in enumerate(STEPS):
                valid=keep&mask[:,h-1,j];y=target[:,h-1,j]
                for a,b,label in [('Direct-MH','MT-Dynamics','MTA'),('Direct-MH-Capacity','MT-Dynamics','CCA')]:
                    diff=np.abs(pred[a][:,k,j]-y)-np.abs(pred[b][:,k,j]-y)
                    point,lo,hi,n,w=patient_boot(diff,sub,valid)
                    rows.append(dict(comparison=label,model_a=a,model_b=b,subset=group,target=var,horizon_seconds=h*10,
                                     difference_mae=point,ci_low=lo,ci_high=hi,patients=n,windows=w))
            # Paired patient-level difference-of-advantages with common 30/300s eligibility.
            valid=keep&mask[:,STEPS[0]-1,j]&mask[:,STEPS[-1]-1,j]
            y30=target[:,STEPS[0]-1,j];y300=target[:,STEPS[-1]-1,j]
            cap=pred['Direct-MH-Capacity'];mt=pred['MT-Dynamics']
            d30=np.abs(cap[:,0,j]-y30)-np.abs(mt[:,0,j]-y30)
            d300=np.abs(cap[:,-1,j]-y300)-np.abs(mt[:,-1,j]-y300)
            point,lo,hi,n,w=patient_boot(d300-d30,sub,valid)
            rows.append(dict(comparison='LongShortGain',model_a='Direct-MH-Capacity',model_b='MT-Dynamics',subset=group,target=var,
                             horizon_seconds=300,difference_mae=point,ci_low=lo,ci_high=hi,patients=n,windows=w))
    return pd.DataFrame(rows)

def action_audit(models,ds,store,device,ref,groups,pred):
    donors=ref['donor'];fa=ref['future_action'];normalized=store.normalize_action(fa)
    donor_actions=np.concatenate([normalized,np.ones_like(normalized)],-1).astype(np.float32)
    assert np.all(ref['subject'][np.flatnonzero(donors>=0)]!=ref['subject'][donors[donors>=0]])
    rows=[];actions={}
    for name,model in models.items():
        actions[name]={'true':pred[name]}
        for condition in ('hold','wrong'):
            p,_=infer(model,ds,store,device,condition,donors,donor_actions);actions[name][condition]=p
            print('ACTION',name,condition,flush=True)
        for group in ('Q4','upcoming_large_intervention','initiation','increase'):
            keep=groups[group]&(donors>=0)
            for k,h in enumerate(STEPS):
                if h not in (6,18,30):continue
                for condition,p in actions[name].items():
                    valid=keep&ref['mask'][:,h-1,0]
                    e=np.abs(p[:,k,0]-ref['target'][:,h-1,0])
                    point,lo,hi,n,w=patient_boot(e,ref['subject'],valid)
                    rows.append(dict(model=name,condition=condition,subset=group,horizon_seconds=h*10,target='BIS',
                                     mae=point,ci_low=lo,ci_high=hi,patients=n,windows=w))
    return pd.DataFrame(rows)

def main():
    seed_all(0);store=Store();ds=Windows(store,'test');device='cuda' if torch.cuda.is_available() else 'cpu'
    ref=np.load(SOURCE/'outputs/test_reference.npz')
    assert len(ds)==83198 and np.array_equal(ds.indices[:,0],ref['case']) and np.array_equal(ds.indices[:,1],ref['t'])
    models={}
    for name,cls,path in [('Direct-MH',Direct,DIRECT/'outputs/checkpoints/Direct-MH.pt'),
                          ('Direct-MH-Capacity',DirectCapacity,ROOT/'checkpoints/Direct-MH-Capacity.pt'),
                          ('MT-Dynamics',MTDynamics,ROOT/'checkpoints/MT-Dynamics.pt')]:
        ck=torch.load(path,map_location=device,weights_only=False)
        m=cls().to(device);m.load_state_dict(ck['model']);m.eval();models[name]=m
    predictions={}
    for name,model in models.items():
        p,z=infer(model,ds,store,device,latent=isinstance(model,MTDynamics));predictions[name]=p
        np.save(ARRAYS/f'{name}_true.npy',p)
        if z:
            for key,v in z.items():np.save(ARRAYS/f'test_{key}.npy',v)
        print('PREDICT',name,flush=True)
    old=np.load(DIRECT/'outputs/arrays/Direct-MH_true.npy',mmap_mode='r')
    delta=float(np.nanmax(np.abs(old[:,np.asarray(STEPS)-1]-predictions['Direct-MH'])))
    assert delta<1e-5,('Direct-MH reproduction failed',delta)
    groups=group_masks(ref,store)
    met=metrics(predictions,ref,groups);met.to_csv(OUT/'horizon_metrics.csv',index=False)
    met[met.subset!='overall'].to_csv(OUT/'subgroup_metrics.csv',index=False)
    pair=paired(predictions,ref,groups);pair.to_csv(OUT/'paired_comparisons.csv',index=False)
    pair[pair.comparison=='MTA'].to_csv(OUT/'multitimescale_advantage.csv',index=False)
    pair[pair.comparison.isin(['CCA','LongShortGain'])].to_csv(OUT/'capacity_controlled_advantage.csv',index=False)
    action=action_audit(models,ds,store,device,ref,groups,predictions);action.to_csv(OUT/'action_sensitivity_metrics.csv',index=False)
    (OUT/'data_summary.json').write_text(json.dumps({'cases':495,'patients':493,'case_splits':{k:len(v) for k,v in store.splits.items()},
      'subject_splits':{k:len(v) for k,v in store.subject_splits.items()},'test_windows':len(ds),'test_patients':len(np.unique(ref['subject'])),
      'subgroup_windows':{k:int(v.sum()) for k,v in groups.items()},'direct_reproduction_max_abs_diff':delta,
      'source_checkpoint_sha256':sha(DIRECT/'outputs/checkpoints/Direct-MH.pt'),'source_test_reference_sha256':sha(SOURCE/'outputs/test_reference.npz')},indent=2))
    print('EVALUATION_COMPLETE',flush=True)
if __name__=='__main__':main()
