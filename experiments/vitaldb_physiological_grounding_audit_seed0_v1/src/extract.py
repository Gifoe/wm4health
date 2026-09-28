"""Stream frozen latent trajectories and factual endpoint encodings for every eligible window."""
import hashlib,time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,Dataset
from common import *

class FactualWindows(Dataset):
    def __init__(self,base,step):self.base=base;self.step=step
    def __len__(self):return len(self.base)
    def __getitem__(self,i):
        cid,t=self.base.indices[i];c=self.base.store.cases[int(cid)]
        u=int(t)+self.step
        past=slice(u-H+1,u+1)
        assert past.stop-1==u
        return (np.concatenate([c['sn'][past],c['sm'][past]],-1),
                np.concatenate([c['an'][past],c['am'][past]],-1),c['dn'])

def case_features(store,ds,split):
    n=len(ds);ids=ds.indices
    subject=save_array(split,'subject',(n,),'int32')
    current=save_array(split,'current',(n,3));target=save_array(split,'target',(n,4,3))
    fresh=save_array(split,'fresh',(n,4,3),'bool');current_fresh=save_array(split,'current_fresh',(n,3),'bool')
    ce=save_array(split,'ce',(n,5,2));raw=save_array(split,'raw_history',(n,19))
    drugs=save_array(split,'drug_history',(n,8));action=save_array(split,'future_action_summary',(n,4,8))
    caseid=save_array(split,'caseid',(n,),'int32');anchor=save_array(split,'anchor',(n,),'int32')
    for cid in np.unique(ids[:,0]):
        rows=np.flatnonzero(ids[:,0]==cid);t=ids[rows,1];c=store.cases[int(cid)]
        subject[rows]=int(c['subjectid']);caseid[rows]=cid;anchor[rows]=t
        current[rows]=c['state'][t];current_fresh[rows]=c['state_fresh'][t]
        for k,h in enumerate(HORIZONS):
            target[rows,k]=c['state'][t+h];fresh[rows,k]=c['state_fresh'][t+h]
        ce[rows,0]=c['ce'][t]
        for k,h in enumerate(HORIZONS):ce[rows,k+1]=c['ce'][t+h]
        s=pd.DataFrame(c['sn']);a=pd.DataFrame(c['an'])
        m1=s.rolling(6,min_periods=1).mean().to_numpy(np.float32)
        m5=s.rolling(30,min_periods=1).mean().to_numpy(np.float32)
        sd=s.rolling(30,min_periods=2).std().fillna(0).to_numpy(np.float32)
        lag=s.to_numpy(np.float32)[np.maximum(t-30,0)]
        raw[rows]=np.concatenate([c['sn'][t],m1[t],m5[t],sd[t],c['sn'][t]-lag,
                                  np.broadcast_to(c['dn'],(len(t),4))],1)
        am=a.rolling(30,min_periods=1).mean().to_numpy(np.float32)
        asd=a.rolling(30,min_periods=2).std().fillna(0).to_numpy(np.float32)
        alag=a.to_numpy(np.float32)[np.maximum(t-30,0)]
        drugs[rows]=np.concatenate([c['an'][t],am[t],asd[t],c['an'][t]-alag],1)
        dose=np.nan_to_num(c['action'],nan=0,posinf=0,neginf=0)
        prefix=np.concatenate([np.zeros((1,2),np.float64),np.cumsum(dose,axis=0)],axis=0)
        for k,h in enumerate(HORIZONS):
            total=(prefix[t+h+1]-prefix[t+1]).astype(np.float32)
            last=c['action'][t+h];first=c['action'][t+1]
            mean=total/h
            delta=last-first
            action[rows,k]=np.concatenate([total,mean,last,delta],1)
        print('CASE_FEATURES',split,int(cid),flush=True) if len(rows)>3000 else None
    for x in [subject,current,target,fresh,current_fresh,ce,raw,drugs,action,caseid,anchor]:x.flush()

@torch.no_grad()
def extract_split(model,store,split,device):
    ds=Windows(store,split);n=len(ds);print('EXTRACT',split,n,flush=True)
    z0=save_array(split,'z0',(n,64));roll=save_array(split,'rollout',(n,F,64))
    pred=save_array(split,'prediction',(n,F,2));zt=save_array(split,'ztrue',(n,4,64))
    loader=DataLoader(ds,batch_size=CFG['batch_size'],num_workers=0,pin_memory=True)
    at=0
    for b in loader:
        b=to_device(b,device);p,z,steps=model(b,'true',return_steps=True);count=len(p)
        z0[at:at+count]=z.cpu().numpy();roll[at:at+count]=steps.cpu().numpy()
        pred[at:at+count]=p.cpu().numpy()*store.norm['state_std'][[0,2]]+store.norm['state_mean'][[0,2]]
        at+=count
    for k,h in enumerate(HORIZONS):
        at=0
        for state,act,demo in DataLoader(FactualWindows(ds,h),batch_size=CFG['batch_size'],num_workers=0,pin_memory=True):
            state=state.to(device);act=act.to(device);demo=demo.to(device)
            x=torch.cat([state,act,demo[:,None,:].expand(-1,H,-1)],-1)
            _,hidden=model.encoder(x);z=model.latent(hidden[-1]);count=len(z)
            zt[at:at+count,k]=z.cpu().numpy();at+=count
        assert at==n
        print('FACTUAL',split,h,flush=True)
    for x in [z0,roll,pred,zt]:x.flush()
    case_features(store,ds,split)
    return n

def main():
    seed_all(0);OUT.mkdir(exist_ok=True);ARRAYS.mkdir(exist_ok=True)
    store=Store();device='cuda' if torch.cuda.is_available() else 'cpu'
    path=SOURCE/'outputs/checkpoints/true.pt';ck=torch.load(path,map_location=device,weights_only=False)
    model=RSSM().to(device);model.set_action_stats(store);model.load_state_dict(ck['model']);model.eval()
    counts={split:extract_split(model,store,split,device) for split in ['train','val','test']}
    old=np.load(SOURCE/'outputs/prediction_true.npz')['prediction']
    new=load('test','prediction');delta=float(np.max(np.abs(new-old)))
    assert delta==0.0,('Checkpoint reproduction mismatch',delta)
    write_json('latent_cache_summary.json',{'eligible_windows':counts,'latent_dim':64,
      'all_rollout_steps_saved':30,'factual_horizons':HORIZONS,'ar_prediction_max_abs_difference':delta,
      'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
      'factual_history_last_index':'t+h','factual_future_beyond_anchor_used':False,
      'large_arrays':'server-only outputs/arrays, excluded from Git'})
    write_json('data_summary.json',{'cases':495,'patients':493,'case_splits':{k:len(v) for k,v in store.splits.items()},
      'subject_splits':{k:len(v) for k,v in store.subject_splits.items()},'eligible_windows':counts,
      'test_patients':int(np.unique(load('test','subject')).size),'sample_seconds':10,
      'history_steps':H,'forecast_steps':F,'ce_reference_type':'device-computed TCI'})
    print('EXTRACTION_COMPLETE',flush=True)

if __name__=='__main__':main()
