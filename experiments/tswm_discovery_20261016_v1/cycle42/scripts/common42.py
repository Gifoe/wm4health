"""Ordinary conditional HMM, exact inference, and frozen synthetic fixture."""
import sys,pathlib,os,json,hashlib,itertools,time,math,random
R=pathlib.Path(__file__).resolve().parents[1];B=R.parent
sys.path.insert(0,str(B/'cycle29/scripts'));import runtime_profile
import numpy as np
import torch
from torch import nn
torch.set_num_threads(1)
K=4;T=24;PREFIX=8
SEEN=[(v,s) for v in range(4) for s in range(4) if s in [v,(v+1)%4]]
CELLS=[(v,s) for v in range(4) for s in range(4)]
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def seed(s):random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def dump(p,obj):pathlib.Path(p).write_text(json.dumps(obj,indent=2),encoding='utf-8')

def data_world(w):
 rng=np.random.default_rng(20264200+w)
 A=np.stack([(.55+.05*s)*np.eye(4)+(1-.55-.05*s)*rng.dirichlet(np.full(4,.8),size=4) for s in range(4)])
 permutations=np.stack([rng.permutation(4) for _ in range(4)])
 C=np.stack([.8*np.eye(4)[:,p]+.2/4 for p in permutations])
 def samples(v,s,n):
  z=rng.integers(4,size=n);out=np.empty((n,T),dtype=np.int64)
  for t in range(T):
   u=rng.random(n);out[:,t]=(u[:,None]>np.cumsum(C[v][z],axis=1)).sum(1)
   u=rng.random(n);z=(u[:,None]>np.cumsum(A[s][z],axis=1)).sum(1)
  return out
 arrays={'A':A,'C':C,'permutations':permutations}
 for phase,n,cells in [('train',128,SEEN),('val',64,SEEN),('eval',256,CELLS)]:
  arrays[phase]=np.stack([samples(v,s,n) for v,s in cells])
 z=rng.integers(4,size=32);anchor=np.empty((32,4),dtype=np.int64)
 for v in range(4):anchor[:,v]=(rng.random(32)[:,None]>np.cumsum(C[v][z],axis=1)).sum(1)
 arrays['anchor']=anchor
 return arrays

class Model(nn.Module):
 def __init__(self,obs_mod,dyn_mod):
  super().__init__();self.obs_mod=obs_mod;self.dyn_mod=dyn_mod
  self.emission=nn.Sequential(nn.Linear(8,32),nn.Tanh(),nn.Linear(32,16))
  self.transition=nn.Sequential(nn.Linear(8,32),nn.Tanh(),nn.Linear(32,16))
  for net in [self.emission,self.transition]:
   nn.init.normal_(net[-1].weight,std=.03);nn.init.normal_(net[-1].bias,std=.03)
 def parameters_for(self,v,s):
  v=torch.nn.functional.one_hot(v,4).to(next(self.parameters()).dtype)
  s=torch.nn.functional.one_hot(s,4).to(v.dtype)
  ec=torch.cat([v,torch.zeros_like(s) if self.obs_mod else s],dim=-1)
  ac=torch.cat([torch.zeros_like(v) if self.dyn_mod else v,s],dim=-1)
  eye=torch.eye(4,device=v.device,dtype=v.dtype)
  C=(self.emission(ec).reshape(-1,4,4)+eye).softmax(-1)
  A=(self.transition(ac).reshape(-1,4,4)+2*eye).softmax(-1)
  return A,C
 def active_parameters(self):return sum(p.numel() for p in self.parameters())-128*(self.obs_mod+self.dyn_mod)

def forward_prob(y,A,C,need_at=PREFIX):
 # p(z_t | y_0:t), likelihood increments: no future input enters the filter.
 b=y.shape[0];state=torch.full((b,4),.25,device=y.device,dtype=A.dtype)
 logparts=[];at=None
 for t in range(y.shape[1]):
  if t:state=torch.bmm(state[:,None,:],A).squeeze(1)
  obs=C.gather(2,y[:,t,None,None].expand(-1,4,1)).squeeze(-1)
  u=state*obs;norm=u.sum(-1);state=u/norm[:,None];logparts.append(norm.log())
  if t+1==need_at:at=state
 return torch.stack(logparts,1),at

def anchor_nll(anchor,model):
 v=torch.arange(4,device=anchor.device);s=torch.zeros(4,device=v.device,dtype=v.dtype)
 _,C=model.parameters_for(v,s)
 evidence=torch.ones((anchor.shape[0],4),device=C.device,dtype=C.dtype)*.25
 for view in range(4):evidence=evidence*C[view,:,anchor[:,view]].T
 return -evidence.sum(-1).log().mean()

def train_batch(model,y,vid,sid,anchors,ntrain):
 A,C=model.parameters_for(vid,sid);parts,_=forward_prob(y,A,C)
 # Unbiased estimator of full-data joint negative log likelihood, common scale.
 return (-parts.sum(1).mean()+len(anchors)/ntrain*anchor_nll(anchors,model))/(T+len(anchors)*4/ntrain)

def flatten(arr,cells,n=None):
 if n is not None:arr=arr[:,:n]
 y=arr.reshape(-1,arr.shape[-1]);ids=np.repeat(np.asarray(cells,dtype=np.int64),arr.shape[1],axis=0)
 return y,ids[:,0],ids[:,1]

@torch.no_grad()
def evaluate(model,arr,cells,device):
 ys,v,s=flatten(arr,cells);chunks=[];nextps=[]
 for j in range(0,len(ys),256):
  y=torch.tensor(ys[j:j+256],device=device);v0=torch.tensor(v[j:j+256],device=device);s0=torch.tensor(s[j:j+256],device=device)
  A,C=model.parameters_for(v0,s0);parts,state=forward_prob(y,A,C)
  pred=torch.bmm(torch.bmm(state[:,None,:],A),C).squeeze(1)
  chunks.append((-parts[:,PREFIX:].mean(1)).cpu().numpy());nextps.append(pred.cpu().numpy())
 return np.concatenate(chunks).reshape(arr.shape[:2]),np.concatenate(nextps).reshape(*arr.shape[:2],4)

@torch.no_grad()
def oracle(arr,A,C,cells):
 ys,v,s=flatten(arr,cells);out=[];probs=[]
 for j in range(0,len(ys),256):
  y=torch.tensor(ys[j:j+256]);a=torch.tensor(A[s[j:j+256]],dtype=torch.float64);c=torch.tensor(C[v[j:j+256]],dtype=torch.float64)
  parts,state=forward_prob(y,a,c);out.append((-parts[:,PREFIX:].mean(1)).numpy())
  probs.append(torch.bmm(torch.bmm(state[:,None,:],a),c).squeeze(1).numpy())
 return np.concatenate(out).reshape(arr.shape[:2]),np.concatenate(probs).reshape(*arr.shape[:2],4)
