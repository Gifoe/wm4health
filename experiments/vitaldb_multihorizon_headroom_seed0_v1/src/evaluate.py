"""Frozen recursive and factual-anchor transition audit; paired direct comparisons."""
import hashlib,json,time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,Dataset
from common import *
from direct import Direct

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

class OracleWindows(Dataset):
    """At h, use factual history only through t+h-1 for a local next-step transition."""
    def __init__(self,base,step):self.base=base;self.step=step
    def __len__(self):return len(self.base)
    def __getitem__(self,i):
        cid,t=self.base.indices[i];c=self.base.store.cases[int(cid)]
        u=int(t)+self.step-1;past=slice(u-H+1,u+1)
        assert past.stop-1==u and u+1==int(t)+self.step
        assert u+1<len(c['action']) and u+1<len(c['state'])
        return {'state':np.concatenate([c['sn'][past],c['sm'][past]],-1),
                'action':np.concatenate([c['an'][past],c['am'][past]],-1),
                'demo':c['dn'],'current':c['sn'][u,[0,2]],
                'initial_current':c['sn'][int(t),[0,2]],
                'next_action':np.concatenate([c['an'][u+1],c['am'][u+1]]),
                'anchor_fresh':c['state_fresh'][u,[0,2]].astype(np.float32)}

@torch.no_grad()
def oracle_infer(model,ds,store,device,step):
    loader=DataLoader(OracleWindows(ds,step),batch_size=512,num_workers=0,pin_memory=True)
    preds=[];latent_only=[];fresh=[]
    for b in loader:
        b=to_device(b,device)
        x=torch.cat([b['state'],b['action'],b['demo'][:,None,:].expand(-1,H,-1)],-1)
        _,hidden=model.encoder(x);z=model.latent(hidden[-1])
        z1=model.transition(torch.cat([b['next_action'],b['demo']],-1),z)
        residual=model.decoder(z1)
        preds.append(physical((b['current']+residual).cpu().numpy(),store))
        latent_only.append(physical((b['initial_current']+residual).cpu().numpy(),store))
        fresh.append(b['anchor_fresh'].cpu().numpy())
    return np.concatenate(preds),np.concatenate(latent_only),np.concatenate(fresh).astype(bool)

@torch.no_grad()
def ar_infer(model,ds,store,device):
    loader=DataLoader(ds,batch_size=512,num_workers=0,pin_memory=True)
    pred=np.empty((len(ds),F,2),np.float32)
    lat=np.lib.format.open_memmap(ARRAYS/'ar_rollout_latents.npy',mode='w+',dtype='float32',shape=(len(ds),F,64))
    offset=0
    for b in loader:
        b=to_device(b,device);p,_,z=model(b,'true',return_steps=True)
        count=len(p);pred[offset:offset+count]=physical(p.cpu().numpy(),store)
        lat[offset:offset+count]=z.cpu().numpy();offset+=count
    lat.flush();return pred

@torch.no_grad()
def direct_infer(model,ds,store,device,steps,condition,donors,donor_actions):
    loader=DataLoader(ds,batch_size=512,num_workers=0,pin_memory=True)
    prediction=np.full((len(ds),F,2),np.nan,np.float32)
    offset=0
    for b in loader:
        count=len(b['index']);ids=b['index'].numpy()
        if condition=='hold':b['future_action']=b['action'][:,-1:].expand(-1,F,-1).clone()
        elif condition=='wrong':
            action=b['future_action'].numpy().copy();valid=donors[ids]>=0
            action[valid]=donor_actions[donors[ids[valid]]]
            b['future_action']=torch.from_numpy(action)
        b=to_device(b,device);p=model(b,steps).cpu().numpy()
        if steps is None:prediction[offset:offset+count]=physical(p,store)
        else:
            v=physical(p,store)
            for k,step in enumerate(steps):prediction[offset:offset+count,step-1]=v[:,k]
        offset+=count
    return prediction

def subsets(ref,store):
    n=len(ref['subject']);up=ref['upcoming_label'];historical=ref['large_label']
    tail=json.loads((SOURCE/'outputs/pump_rate_tail_protocol.json').read_text())
    cap=np.asarray(tail['0.99']['maximum_ml_per_10s'])
    cases=ref['case'];anchors=ref['t']
    current=np.asarray([store.cases[int(c)]['action'][int(t)] for c,t in zip(cases,anchors)])
    clean=(ref['future_action']<=cap).all((1,2))&(current<=cap).all(1)
    result={'overall':np.ones(n,bool),'Q4':ref['quartile']==4,
            'stable_action_Q1':ref['quartile']==1,
            'upcoming_large_intervention':up>0,'historical_large_transition':historical>0,
            'high_rate_tail':~clean,'non_tail':clean}
    for code,name in [(1,'initiation'),(2,'increase'),(3,'decrease'),(4,'stop')]:
        result[name]=up==code
    return result

def patient_stat(values,subject,valid,metric='mae'):
    good=valid&np.isfinite(values)
    if not good.any():return np.nan,0,0
    df=pd.DataFrame({'subject':subject[good],'v':values[good]})
    p=df.groupby('subject').v.mean().to_numpy()
    return (float(np.sqrt(p.mean())) if metric=='rmse' else float(p.mean()),len(p),int(good.sum()))

def bootstrap(values,subject,valid,reps=1000):
    good=valid&np.isfinite(values)
    if not good.any():return np.nan,np.nan,np.nan,0,0
    p=pd.DataFrame({'s':subject[good],'v':values[good]}).groupby('s').v.mean().to_numpy()
    if len(p)==1:return float(p[0]),np.nan,np.nan,1,int(good.sum())
    rng=np.random.default_rng(0)
    draws=p[rng.integers(len(p),size=(reps,len(p)))].mean(1)
    return float(p.mean()),float(np.quantile(draws,.025)),float(np.quantile(draws,.975)),len(p),int(good.sum())

def main():
    seed_all(0);OUT.mkdir(exist_ok=True);ARRAYS.mkdir(exist_ok=True)
    store=Store();ds=Windows(store,'test');ref=np.load(SOURCE/'outputs/test_reference.npz')
    assert len(ds)==83198 and np.array_equal(ds.indices[:,0],ref['case']) and np.array_equal(ds.indices[:,1],ref['t'])
    device='cuda' if torch.cuda.is_available() else 'cpu'
    ckpath=SOURCE/'outputs/checkpoints/true.pt';ck=torch.load(ckpath,map_location=device,weights_only=False)
    ar=RSSM().to(device);ar.set_action_stats(store);ar.load_state_dict(ck['model']);ar.eval()
    ar_pred=ar_infer(ar,ds,store,device)
    original=np.load(SOURCE/'outputs/prediction_true.npz')['prediction']
    delta=float(np.max(np.abs(ar_pred-original)))
    assert delta<1e-5,('Existing checkpoint reproduction failed',delta)
    np.save(ARRAYS/'ar_prediction.npy',ar_pred)
    write_json('ar_reproduction.json',{'checkpoint_sha256':digest(ckpath),'prediction_max_abs_difference':delta,
              'prediction_shape':list(ar_pred.shape),'rollout_latents_shape':[len(ds),F,64]})
    oracle=np.full_like(ar_pred,np.nan);oracle_latent=np.full_like(ar_pred,np.nan)
    oracle_fresh=np.zeros_like(ar_pred,bool)
    for step in HORIZONS:
        p,pl,m=oracle_infer(ar,ds,store,device,step)
        oracle[:,step-1]=p;oracle_latent[:,step-1]=pl;oracle_fresh[:,step-1]=m
        print('ORACLE',step,flush=True)
    np.save(ARRAYS/'oracle_prediction.npy',oracle);np.save(ARRAYS/'oracle_latent_reset_prediction.npy',oracle_latent)
    np.save(ARRAYS/'oracle_anchor_fresh.npy',oracle_fresh)
    donors=ref['donor'];fa=ref['future_action'];an=store.normalize_action(fa)
    donor_actions=np.concatenate([an,np.ones_like(an)],axis=-1).astype(np.float32)
    predictions={'AR-RSSM':ar_pred,'Oracle-State':oracle,'Oracle-Latent-Reset':oracle_latent};corrupt={}
    for name,steps in [('Direct-MH',HORIZONS),('Direct-Traj',None)]:
        model=Direct().to(device)
        checkpoint=torch.load(OUT/'checkpoints'/f'{name}.pt',map_location=device,weights_only=False)
        model.load_state_dict(checkpoint['model']);model.eval()
        for condition in ('true','hold','wrong'):
            p=direct_infer(model,ds,store,device,steps,condition,donors,donor_actions)
            np.save(ARRAYS/f'{name}_{condition}.npy',p)
            if condition=='true':predictions[name]=p
            else:corrupt[(name,condition)]=p
            print('DIRECT',name,condition,flush=True)
    target=ref['target'];mask=ref['mask'].astype(bool);subject=ref['subject']
    groups=subsets(ref,store);rows=[];pairs=[];growth=[];action=[]
    for subset,subset_mask in groups.items():
        for name,p in predictions.items():
            for j,tgt in enumerate(['BIS','MAP']):
                for step in HORIZONS+([0] if name in ('AR-RSSM','Direct-Traj') else []):
                    if step:
                        y=p[:,step-1,j];truth=target[:,step-1,j];valid=mask[:,step-1,j]&subset_mask
                        if name.startswith('Oracle-'):valid=valid&oracle_fresh[:,step-1,j]
                    else:
                        y=p[:,:,j];truth=target[:,:,j]
                        error=np.where(mask[:,:,j],np.abs(y-truth),np.nan)
                        sq=np.where(mask[:,:,j],(y-truth)**2,np.nan)
                        with np.errstate(all='ignore'):
                            e=np.nanmean(error,axis=1);s=np.nanmean(sq,axis=1)
                        valid=subset_mask&np.isfinite(e)
                    if step:e=np.abs(y-truth);s=(y-truth)**2
                    mae,n,w=patient_stat(e,subject,valid)
                    mse,_,_=patient_stat(s,subject,valid)
                    rows.append({'subset':subset,'model':name,'target':tgt,'horizon_seconds':step*10 if step else 0,
                                 'mae':mae,'rmse':np.sqrt(mse),'patients':n,'windows':w})
        for step in HORIZONS:
            for j,tgt in enumerate(['BIS','MAP']):
                truth=target[:,step-1,j];v=mask[:,step-1,j]&subset_mask
                ar_e=np.abs(predictions['AR-RSSM'][:,step-1,j]-truth)
                for name,label in [('Direct-MH','DHA'),('Direct-Traj','DTA'),('Oracle-State','RP'),
                                   ('Oracle-Latent-Reset','RP_latent_reset')]:
                    valid=v.copy()
                    if name.startswith('Oracle-'):valid&=oracle_fresh[:,step-1,j]
                    other=np.abs(predictions[name][:,step-1,j]-truth)
                    value,lo,hi,n,w=bootstrap(ar_e-other,subject,valid,CFG['bootstrap_replicates'])
                    pairs.append({'subset':subset,'target':tgt,'comparison':label,'horizon_seconds':step*10,
                                  'difference_mae':value,'ci_low':lo,'ci_high':hi,'patients':n,'windows':w})
        for name in ('Direct-MH','Direct-Traj'):
            p=predictions[name]
            for condition in ('hold','wrong'):
                q=corrupt[name,condition]
                for step in HORIZONS:
                    for j,tgt in enumerate(['BIS','MAP']):
                        truth=target[:,step-1,j]
                        valid=subset_mask&mask[:,step-1,j]
                        if condition=='wrong':valid&=donors>=0
                        d=np.abs(q[:,step-1,j]-truth)-np.abs(p[:,step-1,j]-truth)
                        value,lo,hi,n,w=bootstrap(d,subject,valid,CFG['bootstrap_replicates'])
                        action.append({'model':name,'subset':subset,'target':tgt,'condition':condition,
                                       'horizon_seconds':step*10,'degradation_mae':value,'ci_low':lo,
                                       'ci_high':hi,'patients':n,'windows':w})
    metric=pd.DataFrame(rows);paired=pd.DataFrame(pairs)
    metric.to_csv(OUT/'horizon_metrics.csv',index=False)
    paired.to_csv(OUT/'paired_comparisons.csv',index=False)
    paired[paired.comparison.isin(['DHA','DTA'])].to_csv(OUT/'direct_horizon_advantage.csv',index=False)
    paired[paired.comparison=='RP'].to_csv(OUT/'recursion_penalty.csv',index=False)
    metric[metric.subset!='overall'].to_csv(OUT/'subgroup_metrics.csv',index=False)
    pd.DataFrame(action).to_csv(OUT/'action_sensitivity_metrics.csv',index=False)
    for (subset,name,tgt),g in metric[(metric.horizon_seconds>0)&metric.horizon_seconds.isin([30,300])].groupby(['subset','model','target']):
        if len(g)!=2:continue
        g=g.set_index('horizon_seconds');a=g.loc[30,'mae'];b=g.loc[300,'mae']
        growth.append({'subset':subset,'model':name,'target':tgt,'mae_30s':a,'mae_300s':b,
                       'absolute_growth':b-a,'normalized_growth':(b-a)/a})
    pd.DataFrame(growth).to_csv(OUT/'error_growth.csv',index=False)
    write_json('data_summary.json',{'cases':495,'patients':493,'split_cases':{k:len(v) for k,v in store.splits.items()},
                'split_subjects':{k:len(v) for k,v in store.subject_splits.items()},'test_windows':len(ds),
                'effective_test_patients':len(np.unique(subject)),'subgroup_windows':{k:int(v.sum()) for k,v in groups.items()},
                'source_round2_checkpoint_sha256':digest(ckpath),'source_test_reference_sha256':digest(SOURCE/'outputs/test_reference.npz')})
    write_json('oracle_state_protocol.json',{'name':'Oracle-State Transition Audit','steps':HORIZONS,
              'factual_anchor_for_step_h':'t+h-1','history_last_index':'t+h-1','next_action_index':'t+h',
              'predicted_target_index':'t+h','future_BIS_or_MAP_beyond_anchor_in_input':False,
              'non_deployable':True,'important_limit':'RP includes the value of factual state at the later anchor; it is not an isolated latent-recursion effect',
              'latent_reset_control':'replaces only factual latent while retaining original t BIS/MAP residual baseline',
              'eligibility':'factual anchor and target must have fresh measurement for each endpoint'})
    print('EVALUATION_COMPLETE',flush=True)

if __name__=='__main__':main()
