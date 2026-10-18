from common65 import *
import sys,copy,numpy as np,torch,torch.nn as nn
sys.path.insert(0,str(ROOT/'upstream'/'csdi'))
from main_model import CSDI_base
H=8;SEEDS=[2021,2022,2023];MECHANISMS=['linear','nonlinear']
def config():return dict(model=dict(timeemb=16,featureemb=8,is_unconditional=0,target_strategy='random'),diffusion=dict(layers=2,channels=32,nheads=4,diffusion_embedding_dim=32,beta_start=.0001,beta_end=.5,num_steps=50,schedule='quad',is_linear=False))
def csdi(device):return CSDI_base(2,copy.deepcopy(config()),device).to(device)
def drift(y,u,mechanism):return .8*y+.7*u if mechanism=='linear' else .65*y+.4*np.tanh(2*y)+.6*u
def sigma(mechanism):return .5 if mechanism=='linear' else .35
def generate(N,mechanism,seed,split):
 rng=np.random.default_rng(np.random.SeedSequence([65000,seed,MECHANISMS.index(mechanism),split]));y=np.empty((N,H));u=np.empty_like(y);last=np.full(N,.25)
 for t in range(H):u[:,t]=-.75*last+.5*rng.normal(size=N);last=drift(last,u[:,t],mechanism)+sigma(mechanism)*rng.normal(size=N);y[:,t]=last
 return np.stack([y,u],axis=1).astype('float32')
def mask(x,kind,rng=None):
 B,K,L=x.shape;m=torch.zeros_like(x)
 if kind=='block':m[:,1]=1
 elif kind=='prefix':
  cuts=(np.random if rng is None else rng).randint(0,L,size=B) if rng is None else rng.integers(0,L,size=B)
  for j,t in enumerate(cuts):m[j,0,:t]=1;m[j,1,:t+1]=1
 else:raise ValueError(kind)
 return m
def side(model,x,m):return model.get_side_info(torch.arange(H,device=x.device).float().expand(len(x),H),m)
class MDN(nn.Module):
 def __init__(self):super().__init__();self.net=nn.Sequential(nn.Linear(2,224),nn.SiLU(),nn.Linear(224,224),nn.SiLU(),nn.Linear(224,9))
 def pars(self,x):
  z=self.net(x);logits,means,scales=z.chunk(3,-1);return torch.log_softmax(logits,-1),means,nn.functional.softplus(scales)+.02
 def nll(self,x,y):
  lp,m,sd=self.pars(x);l=-.5*((y[...,None]-m)/sd)**2-torch.log(sd)-.5*np.log(2*np.pi);return -torch.logsumexp(lp+l,-1).mean()
 def sample(self,x,generator=None):
  lp,m,sd=self.pars(x);idx=torch.multinomial(lp.exp(),1,generator=generator);return m.gather(1,idx).squeeze(-1)+sd.gather(1,idx).squeeze(-1)*torch.randn(len(x),device=x.device,generator=generator)
def transitions(raw):
 y,u=raw[:,0],raw[:,1];prev=np.c_[np.full(len(y),.25,dtype='float32'),y[:,:-1]];return np.stack([prev,u],-1).reshape(-1,2),y.reshape(-1)
