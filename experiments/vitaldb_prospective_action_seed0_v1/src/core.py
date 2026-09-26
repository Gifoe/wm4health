"""Round-1 patient arrays, identical RSSM architecture, distinct action rollout policies."""
import json,random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
import yaml

ROOT=Path(__file__).resolve().parents[1]
CFG=yaml.safe_load((ROOT/'config.yaml').read_text())
SOURCE=(ROOT/CFG['source_experiment']).resolve()
H=CFG['history_steps']; F=CFG['horizon_steps']

def seed_all(seed=0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(6); torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True

class Store:
    def __init__(self):
        self.splits=json.loads((SOURCE/'outputs/split_caseids.json').read_text())
        self.subject_splits=json.loads((SOURCE/'outputs/split_subjectids.json').read_text())
        self.norm={k:np.asarray(v,np.float32) for k,v in json.loads((SOURCE/'outputs/normalization.json').read_text()).items()}
        self.cases={}
        for ids in self.splits.values():
            for cid in ids:
                with np.load(SOURCE/f'data/cases/{cid}.npz') as z:
                    c={k:z[k] for k in z.files}
                s=c['state']; a=c['action']
                c['sm']=np.isfinite(s).astype(np.float32)
                c['am']=np.isfinite(a).astype(np.float32)
                c['sn']=np.nan_to_num((s-self.norm['state_mean'])/self.norm['state_std']).astype(np.float32)
                c['an']=self.normalize_action(a)
                c['dn']=((c['demographics']-self.norm['demo_mean'])/self.norm['demo_std']).astype(np.float32)
                self.cases[cid]=c
        assert len(self.cases)==495

    def normalize_action(self,a):
        return np.nan_to_num((np.log1p(10*np.maximum(a,0))-self.norm['action_mean'])/self.norm['action_std']).astype(np.float32)

    def zero_action(self,shape):
        z=self.normalize_action(np.zeros(shape,np.float32))
        return np.concatenate([z,np.ones(shape,np.float32)],-1)

class Windows(Dataset):
    def __init__(self,store,split,stride=1):
        self.store=store; pairs=[]; original=0; omitted=0
        for cid in store.splits[split]:
            c=store.cases[cid]
            for t in c['anchors']:
                original+=1
                if not np.isfinite(c['action'][t+1:t+F+1]).all():
                    omitted+=1; continue
                pairs.append((cid,int(t)))
        self.indices=np.asarray(pairs,np.int32).reshape(-1,2)[::stride]
        self.original_windows=original; self.omitted_future_action=omitted

    def __len__(self): return len(self.indices)

    def __getitem__(self,i):
        cid,t=self.indices[i]; c=self.store.cases[int(cid)]
        past=slice(t-H+1,t+1); fut=slice(t+1,t+F+1)
        return {'state':np.concatenate([c['sn'][past],c['sm'][past]],-1),
                'action':np.concatenate([c['an'][past],c['am'][past]],-1),
                'future_action':np.concatenate([c['an'][fut],c['am'][fut]],-1),
                'demo':c['dn'], 'current':c['sn'][t,[0,2]],
                'target':c['sn'][fut][:,[0,2]],
                'target_mask':c['state_fresh'][fut][:,[0,2]].astype(np.float32),
                'index':np.int64(i)}

    def reference(self):
        ids=self.indices
        return {'case':ids[:,0], 't':ids[:,1],
                'subject':np.array([int(self.store.cases[int(cid)]['subjectid']) for cid,t in ids]),
                'current':np.array([self.store.cases[int(cid)]['state'][t] for cid,t in ids]),
                'future_action':np.array([self.store.cases[int(cid)]['action'][t+1:t+F+1] for cid,t in ids],np.float32)}

class RSSM(nn.Module):
    """Exact round-1 parameterization; only which action enters each GRUCell differs."""
    def __init__(self):
        super().__init__(); d=CFG['model_dim']
        self.encoder=nn.GRU(14,d,2,batch_first=True,dropout=.1)
        self.latent=nn.LayerNorm(d)
        self.transition=nn.GRUCell(8,d)
        self.decoder=nn.Sequential(nn.Linear(d,64),nn.GELU(),nn.Linear(64,2))
        nn.init.zeros_(self.decoder[-1].weight); nn.init.zeros_(self.decoder[-1].bias)

    def forward(self,b,policy='true',return_steps=False):
        x=torch.cat([b['state'],b['action'],b['demo'][:,None,:].expand(-1,H,-1)],-1)
        _,h=self.encoder(x); z=self.latent(h[-1]); h=z
        if policy=='true': actions=b['future_action']
        elif policy=='hold': actions=b['action'][:,-1:,].expand(-1,F,-1)
        elif policy=='zero':
            # Physical zero in the TRAIN normalization; availability remains one.
            means=torch.as_tensor(self._action_mean,device=h.device,dtype=h.dtype)
            std=torch.as_tensor(self._action_std,device=h.device,dtype=h.dtype)
            zeros=(-means/std)[None,None,:].expand(len(h),F,-1)
            actions=torch.cat([zeros,torch.ones_like(zeros)],-1)
        else: raise ValueError(policy)
        ys=[]; zs=[]
        for k in range(F):
            h=self.transition(torch.cat([actions[:,k],b['demo']],-1),h)
            ys.append(b['current']+self.decoder(h))
            if return_steps: zs.append(h)
        pred=torch.stack(ys,1)
        return pred,z,torch.stack(zs,1) if return_steps else None

    def set_action_stats(self,store):
        self._action_mean=store.norm['action_mean']; self._action_std=store.norm['action_std']

def to_device(batch,device):
    return {k:v.to(device,non_blocking=True) for k,v in batch.items()}
