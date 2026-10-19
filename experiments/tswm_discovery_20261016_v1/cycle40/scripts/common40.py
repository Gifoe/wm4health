"""Ordinary baselines under one fixed-tail contract; not a proposed new method."""
import sys,pathlib,math,time,json,hashlib,copy
R=pathlib.Path(__file__).resolve().parents[1];ROOT=R.parent
sys.path.insert(0,str(ROOT/'cycle29/scripts'));import runtime_profile
sys.path.insert(0,str(R/'vendor'));sys.path.insert(0,str(ROOT/'cycle25/upstream/flownp'))
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from attrdictionary import AttrDict
from models.fnp import FNP,comp_posenc
torch.set_num_threads(1)
DEVICE='cuda'
HISTORY=192;HORIZON=96;REVEALS=[0,8,32]
def seed(s):
 np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def data(name):
 z=np.load(R/'assets'/f'{name}.npz');v=torch.tensor(z['normalized'],dtype=torch.float32,device=DEVICE)
 assert v.ndim==1
 return v
def origins(name,split):return np.load(R/'assets'/f'{name}_{split}_origins.npy')
def coordinates(indices,origin):
 # Positions and known hourly phase contain no future observations.
 idx=torch.as_tensor(indices,device=DEVICE,dtype=torch.float32)
 return torch.stack(((idx-origin)/192.,torch.sin(idx*(2*math.pi/24)),torch.cos(idx*(2*math.pi/24))),-1)
def windows(series,ends):
 ends=torch.as_tensor(ends,device=DEVICE,dtype=torch.long)
 idx=ends[:,None]+torch.arange(-HISTORY,HORIZON,device=DEVICE)[None,:]
 return series[idx].unsqueeze(-1),coordinates(idx,ends[:,None])
def fnp_batch(seq,coords,r):
 return AttrDict(xc=coords[:,:HISTORY+r],yc=seq[:,:HISTORY+r],xt=coords[:,HISTORY+r:],yt=seq[:,HISTORY+r:])
def make_fnp():return FNP(3,1,16,128,3,256,4,0.,4,timesteps=32).to(DEVICE)
class ARMixture(nn.Module):
 """GRU latent dynamics, eight-component Gaussian observation innovation."""
 def __init__(self):
  super().__init__();self.gru=nn.GRU(4,256,num_layers=2,batch_first=True);self.head=nn.Sequential(nn.Linear(256,128),nn.ReLU(),nn.Linear(128,24))
 def process(self,values,next_coords,h=None):return self.gru(torch.cat((values,next_coords),-1),h)
 def pars(self,out,previous):
  logits,delta,rawscale=self.head(out).chunk(3,-1)
  return logits,previous+delta,F.softplus(rawscale)+.001
 def loss(self,seq,coords):
  out,_=self.process(seq[:,:-1],coords[:,1:]);a=HISTORY-1
  logits,mu,sd=self.pars(out[:,a:],seq[:,a:-1])
  truth=seq[:,HISTORY:]
  lp=-.5*((truth-mu)/sd)**2-sd.log()-.5*math.log(2*math.pi)
  return -torch.logsumexp(logits.log_softmax(-1)+lp,-1).mean()
 def context(self,seq,coords,r,cache=None):
  n=HISTORY+r
  if cache is None:
   obs=seq[:,:n];nextc=coords[:,1:n+1];h=None
  else:
   prior_r,out,h,prev=cache;obs=seq[:,HISTORY+prior_r:n];nextc=coords[:,HISTORY+prior_r+1:n+1]
   if not obs.shape[1]:return cache
  out,h=self.process(obs,nextc,h)
  return (r,out[:,-1:],h,seq[:,n-1:n])
 def sample_cached(self,seq,coords,r,cache,draws=32):
  _,out,h,prev=cache
  b=seq.shape[0];out=out.repeat(draws,1,1);h=h.repeat(1,draws,1);prev=prev.repeat(draws,1,1)
  paths=[]
  for lead in range(r,HORIZON):
   logits,mu,sd=self.pars(out,prev)
   k=torch.multinomial(logits[:,0].softmax(-1),1)[:,None,:]
   y=mu.gather(-1,k)+sd.gather(-1,k)*torch.randn_like(prev);paths.append(y)
   if lead<HORIZON-1:out,h=self.process(y,coords[:,HISTORY+lead+1:HISTORY+lead+2].repeat(draws,1,1),h)
   prev=y
  return torch.cat(paths,1).reshape(draws,b,HORIZON-r,1)
@torch.no_grad()
def fnp_sample(model,seq,coords,r,draws,steps):
 model.timesteps=steps
 batch=fnp_batch(seq,coords,r)
 # The native sampler returns trajectory samples in .loc; its 0.001 Normal
 # scale is explicitly a backwards-compatibility placeholder, not extra noise.
 return model.predict(batch.xc,batch.yc,batch.xt,num_samples=draws).loc
@torch.no_grad()
def replay_fnp(model,xc,yc,xt,draws):
 # Independent literal-loop replay of the pinned native public predict API.
 batch=xc.shape[0];n=xt.shape[1];x=xc.repeat(draws,1,1);y=yc.repeat(draws,1,1);q=torch.cat((x,xt.repeat(draws,1,1)),1)
 state=torch.randn(draws*batch,q.shape[1],y.shape[-1],device=q.device)
 cx=torch.cat((x,torch.ones_like(x[:,:,:1])),-1)
 for k in range(model.timesteps):
  t=k/model.timesteps;tt=torch.full_like(q[:,:,:1],t);v=model.encode(comp_posenc(model.dim_posenc,cx),y,comp_posenc(model.dim_posenc,torch.cat((q,tt),-1)),state)
  state=state+((1+t*(1-t))*v+.2*math.sqrt(t*(1-t))*torch.randn_like(state))/model.timesteps
 return state[:,x.shape[1]:].reshape(draws,batch,n,y.shape[-1])
def statehash(model):
 h=hashlib.sha256()
 for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def scores(samples,truth):
 """Fair U-statistic scores. [draw,origin,time], [origin,time]."""
 x=np.asarray(samples,dtype=np.float64);y=np.asarray(truth,dtype=np.float64);m=x.shape[0]
 assert m>1 and x.shape[1:]==y.shape and np.isfinite(x).all()
 ordered=np.sort(x,axis=0);coef=(2*np.arange(1,m+1)-m-1)[:,None,None]
 crps=np.mean(np.abs(x-y[None]),axis=0)-np.sum(coef*ordered,axis=0)/(m*(m-1))
 def scalar_crps(z,w):
  z=np.sort(z,axis=0);c=(2*np.arange(1,m+1)-m-1)[:,None]
  return np.abs(z-w).mean(0)-(c*z).sum(0)/(m*(m-1))
 first=np.linalg.norm(x-y[None],axis=-1).mean(0)
 pair=np.zeros(len(y))
 for i in range(m):pair+=np.linalg.norm(x[i+1:]-x[i],axis=-1).sum(0)
 energy=(first-pair/(m*(m-1)))/math.sqrt(y.shape[-1])
 return {'energy':energy,'marginal_crps':crps.mean(-1),'mean_crps':scalar_crps(x.mean(-1),y.mean(-1)),
         'mae':np.abs(x.mean(0)-y).mean(-1),'coverage80':((y>=np.quantile(x,.1,axis=0))&(y<=np.quantile(x,.9,axis=0))).mean(-1)}
def timed(fn):
 torch.cuda.synchronize();a=time.perf_counter();x=fn();torch.cuda.synchronize();return x,time.perf_counter()-a
