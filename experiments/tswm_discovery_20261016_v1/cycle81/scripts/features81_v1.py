import sys,pathlib,copy,time
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common81_v1 import *
sys.path.insert(0,str(PROJECT/'cycle80/scripts'))
import native80_v1 as old
import numpy as np
import torch
from fim.models.ode import load_fim_ode_local
from ode_tutorial_helper import load_odebench_system,make_fim_vf_fn

class NativeFeatures:
 def __init__(self,model,x,t):
  self.model=model;self.x=np.asarray(x).copy();self.t=np.asarray(t).copy();self.m=x.mean(0);self.s=np.maximum(x.std(0),1e-8)
  self.traj=torch.tensor(x[None,None],dtype=torch.float32);self.times=torch.tensor(t[None,None,:,None],dtype=torch.float32);self.mask=torch.ones((*self.traj.shape[:-1],1),dtype=torch.bool)
  with torch.no_grad():self.encoded,self.feature_mask,self.concept=model.trajectory_encoding(self.traj,self.times,self.mask)
  self.linear=model.functional_decoder.projection.layers.output;assert isinstance(self.linear,torch.nn.Linear)
  self.weight=self.linear.weight.detach().numpy().copy();self.bias=self.linear.bias.detach().numpy().copy();self.p=self.linear.in_features+1
  rng=np.random.default_rng(202681);self.freq=rng.normal(size=(2,self.p-1));self.phase=rng.uniform(0,2*np.pi,self.p-1)
 def evaluate(self,x):
  x=np.atleast_2d(x);seen=[]
  def hook(module,args):seen.append(args[0].detach().numpy().copy())
  handle=self.linear.register_forward_pre_hook(hook)
  try:
   with torch.no_grad():
    loc=self.model.pad_if_necessary(torch.tensor(x[None],dtype=torch.float32));norm=self.model.spatial_norm.normalization_map(loc,self.concept._states_norm_stats);out=self.model.function_decoding(norm,self.feature_mask,self.encoded,copy.deepcopy(self.concept));normal=out.predictions.drift.detach().numpy()[0].copy();unit=copy.deepcopy(out.predictions);unit.drift=torch.ones_like(unit.drift);unit.renormalize();factor=unit.drift.detach().numpy()[0].copy();physical=self.model.get_prediction_for_eval(out)[0,:,:2].numpy().copy()
  finally:handle.remove()
  assert len(seen)==1;h=seen[0][0];assert h.ndim==2
  return np.c_[h,np.ones(len(h))],physical,normal,factor
 def base(self,x):return self.evaluate(x)[1]
 def phi(self,x,kind):
  x=np.atleast_2d(x);h,f,_,_=self.evaluate(x)
  if kind.startswith('fim_'):return h,f
  z=(x-self.m)/self.s
  if kind=='rff_residual':return np.c_[np.sqrt(2/(self.p-1))*np.cos(z@self.freq+self.phase),np.ones(len(z))],f
  if kind=='affine_residual':return np.c_[np.ones(len(z)),z,f/self.s],f
  raise ValueError(kind)

def load_assets():
 assets=old.assets();weights=next(a for a in assets if a['relative'].endswith('.safetensors'));data=next(a for a in assets if a['relative'].endswith('strogatz_extended.json'));model=load_fim_ode_local(pathlib.Path(weights['path']).parent,'cpu');model.eval()
 for p in model.parameters():p.requires_grad_(False)
 return model,weights,data

def qualify(model,data,sid):
 tt,xx,_=load_odebench_system(sid,data['path'],noise_sigma=.03,subsample_fraction=.5,rng=np.random.default_rng(2026810));x=xx[0];t=tt[0,:,0];nf=NativeFeatures(model,x[:192],t[:192]);q=x[np.linspace(0,191,17,dtype=int)];h,f,n,g=nf.evaluate(q);affine=h[:,:-1]@nf.weight.T+nf.bias;recovered=affine*g
 native=make_fim_vf_fn(model,x[None,:192],t[None,:192,None],'cpu')(q)
 with torch.no_grad():direct=model.get_prediction_for_eval(model.model_forward(nf.traj,nf.times,torch.tensor(q[None],dtype=torch.float32),nf.mask))[0,:,:2].numpy()
 tests=dict(hidden_readout_max_abs=float(abs(affine-n).max()),physical_affine_max_abs=float(abs(recovered[:,:2]-f).max()),cached_native_max_abs=float(abs(native-f).max()),direct_native_max_abs=float(abs(direct-f).max()),feature_count=nf.p,context_steps=192,qualification_queries=len(q))
 assert np.allclose(affine,n,atol=2e-5,rtol=2e-5) and np.allclose(recovered[:,:2],f,atol=2e-5,rtol=2e-5) and np.allclose(native,f,atol=2e-5,rtol=2e-5) and np.allclose(direct,f,atol=2e-5,rtol=2e-5)
 return tests
