"""Ordinary neural observation/dynamics composition on the author NCDSSM core."""
from joint_adapter46 import *
import numpy as np,json,hashlib,random,time
from torch import nn
HIST=8;FUTURE=16;T=24;Y=8;D=4;SEEDS=[2026461,2026462]
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def dump(p,x):pathlib.Path(p).write_text(json.dumps(x,indent=2),encoding='utf-8')
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8'))
def seed(s):random.seed(s);np.random.seed(s);torch.manual_seed(s)
def checksum(model):
 h=hashlib.sha256()
 for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()
class Context:
 def __init__(self):self.view=0;self.system=0
 @property
 def bits(self):return (float(self.system&1),float((self.system>>1)&1))
def network(a,b,width=64):return nn.Sequential(nn.Linear(a,width),nn.Tanh(),nn.Linear(width,width),nn.Tanh(),nn.Linear(width,b))
def features(x,v,ctx,observation_shared):
 view=x.new_zeros(*x.shape[:-1],4);view[...,v]=1;bits=x.new_tensor((0.,0.) if observation_shared else ctx.bits).expand(*x.shape[:-1],2);return torch.cat((x,view,bits),-1)
class Recognition(nn.Module):
 def __init__(self,ctx,shared):super().__init__();self.ctx=ctx;self.shared=shared;self.net=network(Y+6,2*D)
 def factor(self,y,v):
  p=self.net(features(y,v,self.ctx,self.shared));return p[...,:D],p[...,D:].clamp(-5,1)
 def forward(self,y):
  if self.ctx.view is not None:return torch.cat(self.factor(y,self.ctx.view),-1)
  factors=[self.factor(y[...,v*Y:(v+1)*Y],v) for v in range(4)];precisions=[torch.exp(-2*l) for _,l in factors];total=sum(precisions);mean=sum(m*p for (m,_),p in zip(factors,precisions))/total;return torch.cat((mean,-.5*total.log()),-1)
class Observation(nn.Module):
 def __init__(self,ctx,shared):super().__init__();self.ctx=ctx;self.shared=shared;self.net=network(D+6,2*Y)
 def params(self,a,v):
  p=self.net(features(a,v,self.ctx,self.shared));return p[...,:Y],p[...,Y:].clamp(-5,1).exp()
 def forward(self,a):
  if self.ctx.view is not None:mean,scale=self.params(a,self.ctx.view)
  else:
   p=[self.params(a,v) for v in range(4)];mean=torch.cat([m for m,s in p],-1);scale=torch.cat([s for m,s in p],-1)
  return torch.distributions.Normal(mean,scale)
class Dynamics(nn.Module):
 def __init__(self,ctx,compositional):
  super().__init__();self.ctx=ctx;self.compositional=compositional;self.z_dim=D;self.uQ=nn.Parameter(torch.full((D,),math.log(.2**2)));self.cached=None
  if compositional:
   self.raw_skew=nn.Parameter(torch.randn(3,D,D)*.02);raw=np.log(np.expm1(np.asarray([.349,.009,.009])));self.raw_decay=nn.Parameter(torch.tensor(np.repeat(raw[:,None],D,axis=1),dtype=torch.float32))
  else:
   self.net=nn.Sequential(nn.Linear(2,16),nn.Tanh(),nn.Linear(16,D*D+D));nn.init.normal_(self.net[-1].weight,std=.02);nn.init.zeros_(self.net[-1].bias)
   with torch.no_grad():self.net[-1].bias[D*D:].fill_(float(np.log(np.expm1(.349))))
 @property
 def F(self):
  if self.cached is None:
   if self.compositional:
    matrices=.5*(self.raw_skew-self.raw_skew.transpose(-1,-2))-torch.diag_embed(torch.nn.functional.softplus(self.raw_decay)+.001);self.cached=matrices[0]+self.ctx.bits[0]*matrices[1]+self.ctx.bits[1]*matrices[2]
   else:
    p=self.net(self.uQ.new_tensor(self.ctx.bits));M=p[:D*D].reshape(D,D);self.cached=.5*(M-M.T)-torch.diag(torch.nn.functional.softplus(p[D*D:])+.001)
  return self.cached
 @property
 def Q(self):return torch.diag(self.uQ.exp().clamp_min(1e-4))
class World(nn.Module):
 def __init__(self,observation_shared,dynamics_compositional):
  super().__init__();self.ctx=Context();recognition=Recognition(self.ctx,observation_shared);observation=Observation(self.ctx,observation_shared);aux=AuxInferenceModel(recognition,GaussianOutput(nn.Identity(),dist_dim=D,use_tied_cov=False,use_independent=True),aux_dim=D);self.native=NCDSSMLTI(aux,observation,aux_dim=D,z_dim=D,y_dim=Y,u_dim=0,H=torch.eye(D),integration_method='matrix_exp',R=torch.full((D,),.15**2));self.native.base_ssm.dynamics=Dynamics(self.ctx,dynamics_compositional)
 def set_context(self,view,system):self.ctx.view=view;self.ctx.system=system;self.native.y_dim=Y if view is not None else 4*Y;self.native.base_ssm.dynamics.cached=None
 def loss(self,y,view,system):
  self.set_context(view,system);mask=y.new_ones(y.shape[:2]);times=torch.arange(y.shape[1],device=y.device,dtype=y.dtype)*.25;return -self.native(y,mask,times,num_samples=1)['elbo'].mean()/(y.shape[1]*y.shape[2])
 @torch.no_grad()
 def sample(self,history,view,system,n,draw_seed):
  self.eval();self.set_context(view,system);history=torch.as_tensor(history,device=next(self.parameters()).device,dtype=torch.float32);mask=history.new_ones(history.shape[:2]);past=torch.arange(HIST,device=history.device)*.25;future=torch.arange(HIST,T,device=history.device)*.25;return sample_joint(self.native,history,mask,past,future,n,draw_seed)
def energy(samples,truth):
 truth=torch.as_tensor(truth,device=samples.device,dtype=samples.dtype);flat=samples.flatten(2);n=flat.shape[1];distance=torch.cdist(flat,flat);distance.diagonal(dim1=-2,dim2=-1).zero_();return ((flat-truth.flatten(1)[:,None]).norm(dim=-1).mean(1)-distance.sum((1,2))/(2*n*(n-1)))/math.sqrt(FUTURE*Y)
