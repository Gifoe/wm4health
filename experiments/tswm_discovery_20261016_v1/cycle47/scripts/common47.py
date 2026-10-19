"""Exact finite-grid test of a provisional joint marked-trace objective."""
import pathlib,sys,math,os,json,hashlib,random,ast,datetime,time
R=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(R.parent/'cycle29/scripts'));import runtime_profile
import numpy as np
import torch
from torch import nn
K=65;T=64;HIST=32;FUTURE=32;DT=.125;OBS=.2;SEEDS=[2026471,2026472];CASES=['ou','bistable'];METHODS=['mean_nll','full_nll','full_laplace','full_hybrid'];RHOS=[.5,1.]
SRC=R/'source_archive/HyperNSDE';component_path=SRC/'src/modules/long_latent_model.py';component_text=component_path.read_text(encoding='utf-8');component_tree=ast.parse(component_text);component_nodes=[n for n in component_tree.body if isinstance(n,ast.ClassDef) and n.name in ['LipSwish','ScaledTanh','MLP_standard']];assert len(component_nodes)==3;component_namespace={'torch':torch,'nn':nn};exec(compile(ast.Module(body=component_nodes,type_ignores=[]),str(component_path),'exec'),component_namespace);AuthorMLP=component_namespace['MLP_standard']
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def dump(p,x):pathlib.Path(p).write_text(json.dumps(x,indent=2),encoding='utf-8')
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8'))
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def seed(s):random.seed(s);np.random.seed(s);torch.manual_seed(s)
def checksum(m):
 h=hashlib.sha256()
 for k,v in m.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def binned_normal(interior,mean,scale):
 cdf=torch.special.ndtr((interior-mean[...,None])/scale);values=torch.cat([torch.zeros_like(cdf[...,:1]),cdf,torch.ones_like(cdf[...,:1])],-1);p=values.diff(dim=-1).clamp_min(1e-12);return p/p.sum(-1,keepdim=True)
class World(nn.Module):
 def __init__(self,mean_clock=False):
  super().__init__();self.mean_clock=mean_clock;grid=torch.linspace(-3.5,3.5,K);mid=(grid[:-1]+grid[1:])/2;mark_edges=grid.new_tensor([-1.5,-1.,-.5,0.,.5,1.,1.5]);self.register_buffer('grid',grid);self.register_buffer('mid',mid);self.register_buffer('mark_edges',mark_edges);self.register_buffer('mark',binned_normal(mark_edges,grid,OBS));self.register_buffer('prior',binned_normal(mid,grid.new_tensor(0.),.8));self.fnet=AuthorMLP(1,1,16,1,activation='Tanh');self.rnet=AuthorMLP(1,1,16,1,activation='Tanh');self.raw_decay=nn.Parameter(grid.new_tensor(float(np.log(np.expm1(.45)))));self.raw_sigma=nn.Parameter(grid.new_tensor(math.log(.5/.85)))
  for net in [self.fnet,self.rnet]:nn.init.normal_(net._model[-1].weight,std=.01);nn.init.zeros_(net._model[-1].bias)
  with torch.no_grad():self.rnet._model[-1].bias.fill_(math.log((2.-.05)/(6.-2.)))
 def drift(self,z):return -(torch.nn.functional.softplus(self.raw_decay)+.05)*z+1.5*torch.tanh(self.fnet(z[...,None]).squeeze(-1))
 def rate(self,z):return .05+5.95*torch.sigmoid(self.rnet(z[...,None]).squeeze(-1))
 @property
 def sigma(self):return .15+1.35*torch.sigmoid(self.raw_sigma)
 def components(self):
  f=self.drift(self.grid);P=binned_normal(self.mid,self.grid+DT*f,self.sigma*math.sqrt(DT));return P,self.rate(self.grid)
 def mean_rates(self,P,steps):
  alpha=(self.prior[:,None]*self.mark).T;alpha=alpha/alpha.sum(-1,keepdim=True);means=[]
  for t in range(steps):means.append(alpha@self.grid);alpha=alpha@P
  return self.rate(torch.stack(means,1))
 def joint(self,initial,counts,rho,P=None,rate=None,return_posterior=False):
  if P is None:P,rate=self.components()
  initial=initial.long();B,steps,_=counts.shape;rho=torch.as_tensor(rho,device=self.grid.device,dtype=self.grid.dtype).reshape(-1);rho=rho.expand(B) if rho.numel()==1 else rho;alpha=self.prior[None]*self.mark[:,initial].T;normalizer=alpha.sum(-1);log_likelihood=normalizer.log();alpha=alpha/normalizer[:,None];mark_term=counts@self.mark.log().T;total=counts.sum(-1);constant=(total*(DT*rho[:,None]).log()-torch.lgamma(counts+1).sum(-1));rates=self.mean_rates(P,steps)[initial] if self.mean_clock else rate;logrates=rates.log()
  for t in range(steps):
   if self.mean_clock:log_emission=mark_term[:,t]+(constant[:,t]+total[:,t]*logrates[:,t]-DT*rho*rates[:,t])[:,None]
   else:log_emission=mark_term[:,t]+constant[:,t,None]+total[:,t,None]*logrates[None]-DT*rho[:,None]*rates[None]
   shift=log_emission.max(-1).values;weighted=alpha*(log_emission-shift[:,None]).exp();norm=weighted.sum(-1).clamp_min(1e-30);log_likelihood=log_likelihood+shift+norm.log();alpha=(weighted/norm[:,None])@P
  return (log_likelihood,alpha) if return_posterior else log_likelihood
 def laplace(self,weights,rho,P=None,rate=None):
  assert not self.mean_clock,'No mean-clock Laplace training arm';P,rate=self.components() if P is None else (P,rate);mark_factor=weights.neg().exp()@self.mark.T;alpha=self.prior[None]*mark_factor[:,0]
  for t in range(T):alpha=(alpha*torch.exp(-DT*rho*rate[None]*(1-mark_factor[:,t+1])))@P
  return alpha.sum(-1)
 @torch.no_grad()
 def sample(self,initial,counts,rho,n,draw_seed):
  self.eval();P,rate=self.components();_,alpha=self.joint(initial,counts,rho,P,rate,True);B=len(alpha);generator=torch.Generator(device=self.grid.device).manual_seed(draw_seed);state=torch.multinomial(alpha,n,replacement=True,generator=generator).reshape(-1);values=[]
  for t in range(FUTURE):
   values.append(self.grid[state]+OBS*torch.randn(B*n,device=self.grid.device,dtype=self.grid.dtype,generator=generator))
   if t<FUTURE-1:state=torch.multinomial(P[state],1,generator=generator).squeeze(-1)
  return torch.stack(values,-1).reshape(B,n,FUTURE)
 @torch.no_grad()
 def coupling(self,rho=1.):
  P,rate=self.components();alpha=(self.prior[:,None]*self.mark).T;probability=alpha.sum(-1);alpha/=probability[:,None];mixed=alpha.new_zeros(alpha.shape);mean_rates=self.mean_rates(P,HIST) if self.mean_clock else None
  for t in range(HIST):
   rates=mean_rates[:,t,None] if self.mean_clock else rate[None];mixed=(mixed+alpha*(DT*rho*rates))@P;alpha=alpha@P
  cov=mixed@self.grid-mixed.sum(-1)*(alpha@self.grid);return probability,cov

def energy(samples,truth):
 flat=samples;target=torch.as_tensor(truth,device=samples.device,dtype=samples.dtype);n=flat.shape[1];dist=torch.cdist(flat,flat,compute_mode='donot_use_mm_for_euclid_dist');dist.diagonal(dim1=-2,dim2=-1).zero_();return ((flat-target[:,None]).norm(dim=-1).mean(1)-dist.sum((1,2))/(2*n*(n-1)))/math.sqrt(FUTURE)
def feature_bank(n,draw_seed):
 gen=np.random.default_rng(draw_seed);weights=np.zeros((n,T+1,8),dtype='float32')
 def selected():
  choice=gen.integers(3)
  if choice==0:return np.ones(8,dtype=bool)
  if choice==1:return np.arange(8)>=4 if gen.random()<.5 else np.arange(8)<4
  return np.arange(8)==gen.integers(8)
 for i in range(n):
  if gen.random()<.5:weights[i,0,selected()]=float(gen.uniform(.1,1.5))
  for _ in range(1+int(gen.random()<.5)):
   length=int(gen.choice([4,8,16]));start=int(gen.integers(T-length+1));weights[i,1+start:1+start+length,selected()]+=float(gen.uniform(.1,1.5))
 return weights
def empirical_features(initial,counts,weights):
 # No hidden-state labels. The initial measurement is a scheduled bin, not a Cox count.
 return np.exp(-weights[:,0,initial].T-np.einsum('btm,ftm->bf',counts,weights[:,1:],optimize=True))
def objective(model,method,initial,counts,rho,weights,mu,var,feature_rho):
 P,rate=model.components();parts={}
 if method in ['mean_nll','full_nll','full_hybrid']:parts['nll']=-model.joint(initial,counts,rho,P,rate).mean()/(T+1)
 if method in ['full_laplace','full_hybrid']:parts['laplace']=10*((model.laplace(weights,feature_rho,P,rate)-mu).square()/(var+.02)).mean()
 return sum(parts.values()),parts
