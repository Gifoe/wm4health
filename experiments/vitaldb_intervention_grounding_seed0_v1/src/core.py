"""Features, dynamic windows, matched-capacity baselines. CE never enters features."""
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
import yaml

ROOT=Path(__file__).resolve().parents[1]
CFG=yaml.safe_load((ROOT/'config.yaml').read_text())

def seed_all(seed=0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(6)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True

class Store:
    def __init__(self):
        self.splits=json.loads((ROOT/'outputs/split_caseids.json').read_text())
        self.cases={}
        for split in self.splits.values():
            for cid in split:
                with np.load(ROOT/f'data/cases/{cid}.npz') as d: self.cases[cid]={k:d[k] for k in d.files}
        stats_path=ROOT/'outputs/normalization.json'
        if stats_path.exists(): self.stats=json.loads(stats_path.read_text())
        else:
            train=[self.cases[c] for c in self.splits['train']]
            state=np.concatenate([c['state'] for c in train])
            action=np.log1p(10*np.concatenate([c['action'] for c in train]))
            demo=np.stack([c['demographics'] for c in train])
            self.stats={}
            for name,a in [('state',state),('action',action),('demo',demo)]:
                self.stats[name+'_mean']=np.nanmean(a,0).tolist()
                self.stats[name+'_std']=np.maximum(np.nanstd(a,0),1e-3).tolist()
            stats_path.write_text(json.dumps(self.stats,indent=2))
        self.norm={k:np.array(v,np.float32) for k,v in self.stats.items()}
        for c in self.cases.values():
            state=c['state']; action=c['action']
            c['state_mask']=np.isfinite(state).astype(np.float32)
            c['action_mask']=np.isfinite(action).astype(np.float32)
            c['sn']=np.nan_to_num((state-self.norm['state_mean'])/self.norm['state_std']).astype(np.float32)
            c['an']=self.normalize_action(action)
            c['dn']=((c['demographics']-self.norm['demo_mean'])/self.norm['demo_std']).astype(np.float32)
        self.make_events()

    def normalize_action(self,a):
        return np.nan_to_num((np.log1p(10*np.maximum(a,0))-self.norm['action_mean'])/self.norm['action_std']).astype(np.float32)

    def make_events(self):
        """Train-only magnitude thresholds; sustained pre/post one-minute medians."""
        from numpy.lib.stride_tricks import sliding_window_view
        summaries={}
        for cid,c in self.cases.items():
            a=c['action']; n=len(a)
            med=np.full_like(a,np.nan)
            med[5:]=np.nanmedian(sliding_window_view(a,6,axis=0),axis=-1)
            pre=np.full_like(a,np.nan); pre[6:]=med[:-6]
            summaries[cid]=(pre,med)
        threshold_path=ROOT/'outputs/event_thresholds.json'
        if threshold_path.exists(): thresholds=json.loads(threshold_path.read_text())
        else:
            thresholds=[]
            for j in range(2):
                rates=np.concatenate([self.cases[c]['action'][:,j] for c in self.splits['train']])
                pos=rates[np.isfinite(rates)&(rates>0)]
                # Positive-rate lower decile defines effective-on; upper quartile
                # of nonzero sustained changes defines a substantial change.
                change=np.concatenate([np.abs(summaries[c][1][:,j]-summaries[c][0][:,j]) for c in self.splits['train']])
                change=change[np.isfinite(change)&(change>1e-6)]
                thresholds.append({'on':float(np.quantile(pos,.1)), 'change':float(np.quantile(change,.75))})
            threshold_path.write_text(json.dumps(thresholds,indent=2))
        self.thresholds=thresholds
        event_records=[]
        for cid,c in self.cases.items():
            pre,post=summaries[cid]; n=len(post)
            tag=np.zeros(n,np.int8); events=[]
            for j in range(2):
                on=thresholds[j]['on']; change=thresholds[j]['change']
                labels=np.zeros(n,np.int8)
                finite=np.isfinite(pre[:,j])&np.isfinite(post[:,j])
                cs=np.r_[0,np.cumsum(np.isfinite(c['action'][:,j]))]
                complete=np.zeros(n,bool)
                complete[11:]=(cs[12:]-cs[:-12])==12
                finite &= complete
                labels[finite&(pre[:,j]<=1e-6)&(post[:,j]>on)]=1
                labels[finite&(pre[:,j]>on)&(post[:,j]<=1e-6)]=4
                ongoing=finite&(pre[:,j]>1e-6)&(post[:,j]>1e-6)
                labels[ongoing&(post[:,j]-pre[:,j]>=change)]=2
                labels[ongoing&(pre[:,j]-post[:,j]>=change)]=3
                last=-100
                for t in np.flatnonzero(labels):
                    if t-last<12: continue
                    last=t
                    # t is the end of the six samples used to confirm the change.
                    # All action values used for labeling are observed by t.
                    tag[t:min(n,t+13)]=labels[t]
                    row={'caseid':cid,'subjectid':int(c['subjectid']),'confirmed_index':int(t),
                         'confirmed_seconds':float(c['time'][t]),'drug':j,
                         'kind':['none','initiation','increase','decrease','stop'][labels[t]],
                         'pre_ml_per_10s':float(pre[t,j]),'post_ml_per_10s':float(post[t,j])}
                    event_records.append(row); events.append(row)
            c['event']=tag; c['events']=events
        self.event_records=event_records

class Windows(Dataset):
    def __init__(self,store,split,stride=1,indices=None):
        self.store=store
        if indices is None:
            pairs=[]
            for cid in store.splits[split]:
                ts=store.cases[cid]['anchors'][::stride]
                pairs.extend((cid,int(t)) for t in ts)
            self.indices=np.array(pairs,np.int32)
        else: self.indices=np.asarray(indices,np.int32)

    def __len__(self): return len(self.indices)

    def __getitem__(self,i):
        cid,t=self.indices[i]; c=self.store.cases[int(cid)]; h=180; f=30
        sl=slice(t-h+1,t+1); fut=slice(t+1,t+f+1)
        return {'state':np.concatenate([c['sn'][sl],c['state_mask'][sl]],axis=-1),
                'action':np.concatenate([c['an'][sl],c['action_mask'][sl]],axis=-1),
                'demo':c['dn'],'target':c['sn'][fut][:,[0,2]],
                'target_mask':c['state_fresh'][fut][:,[0,2]].astype(np.float32),
                'current':c['sn'][t,[0,2]],'index':np.int64(i)}

    def reference(self):
        ce=[]; subjects=[]; event=[]; current=[]
        for cid,t in self.indices:
            c=self.store.cases[int(cid)]
            ce.append(c['ce'][t]); subjects.append(int(c['subjectid'])); event.append(int(c['event'][t])); current.append(c['state'][t])
        return {'ce':np.array(ce),'subject':np.array(subjects),'event':np.array(event),
                'current':np.array(current),'case':self.indices[:,0],'t':self.indices[:,1]}

class TransformerModel(nn.Module):
    def __init__(self,action=True):
        super().__init__(); self.use_action=action; d=CFG['model_dim']
        self.project=nn.Linear(14 if action else 10,d)
        self.position=nn.Parameter(torch.randn(1,180,d)*.02)
        layer=nn.TransformerEncoderLayer(d,4,128,dropout=.1,batch_first=True,norm_first=True,activation='gelu')
        self.encoder=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.latent=nn.Sequential(nn.Linear(2*d,d),nn.LayerNorm(d),nn.GELU())
        self.head=nn.Sequential(nn.Linear(d,128),nn.GELU(),nn.Linear(128,60))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)

    def forward(self,b):
        parts=[b['state'],b['demo'][:,None,:].expand(-1,180,-1)]
        if self.use_action: parts.append(b['action'])
        x=self.encoder(self.project(torch.cat(parts,-1))+self.position)
        z=self.latent(torch.cat([x[:,-1],x.mean(1)],-1))
        y=b['current'][:,None,:]+self.head(z).view(-1,30,2)
        return y,z

class RSSMModel(nn.Module):
    """Compact deterministic RSSM-style state-space baseline, not stochastic Dreamer."""
    def __init__(self):
        super().__init__(); d=CFG['model_dim']
        self.encoder=nn.GRU(14,d,2,batch_first=True,dropout=.1)
        self.latent=nn.LayerNorm(d)
        self.transition=nn.GRUCell(8,d)
        self.decoder=nn.Sequential(nn.Linear(d,64),nn.GELU(),nn.Linear(64,2))
        nn.init.zeros_(self.decoder[-1].weight); nn.init.zeros_(self.decoder[-1].bias)

    def forward(self,b):
        x=torch.cat([b['state'],b['action'],b['demo'][:,None,:].expand(-1,180,-1)],-1)
        _,h=self.encoder(x); z=self.latent(h[-1]); h=z
        a=torch.cat([b['action'][:,-1],b['demo']],-1)
        ys=[]
        for _ in range(30):
            h=self.transition(a,h)
            ys.append(b['current']+self.decoder(h))
        return torch.stack(ys,1),z

def model_for(name):
    if name=='State-only': return TransformerModel(False)
    if name=='Action Transformer': return TransformerModel(True)
    if name=='RSSM': return RSSMModel()
    raise ValueError(name)

MODEL_NAMES=['State-only','Action Transformer','RSSM']

def filename(name): return name.lower().replace(' ','_').replace('-','_')

def to_device(batch,device):
    return {k:v.to(device,non_blocking=True) for k,v in batch.items()}

def perturb_actions(batch,store,shift=0,scale=1.):
    out=dict(batch); a=batch['action'].clone()
    if shift:
        # Positive delta delays actions; negative delta advances within observed
        # history. Edge replication never reads future actions and never wraps.
        ids=(torch.arange(180,device=a.device)-int(shift/10)).clamp(0,179)
        a=a[:,ids]
    if scale!=1:
        mean=torch.tensor(store.norm['action_mean'],device=a.device)
        std=torch.tensor(store.norm['action_std'],device=a.device)
        raw=torch.expm1(a[:,:,:2]*std+mean).clamp_min(0)/10
        a[:,:,:2]=(torch.log1p(10*raw*scale)-mean)/std
        a[:,:,:2]*=a[:,:,2:]
    out['action']=a
    return out
