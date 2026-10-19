"""Ordinary fitted-generative-law sampler, not native marginal-time forecasts."""
import pathlib,sys,math
R=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(R.parent/'cycle29/scripts'));import runtime_profile
import torch
SRC=R.parent/'cycle45/source_archive/ncdssm';sys.path.insert(0,str(SRC/'src'))
from ncdssm.models import NCDSSMLTI,NCDSSMNL
from ncdssm.models.components import AuxInferenceModel,GaussianOutput
from ncdssm.modules import MLP

@torch.no_grad()
def state_paths(base,mean,chol,last_time,future_times,generator,step_size=.0125):
 state=mean+torch.bmm(chol,torch.randn(mean.shape,device=mean.device,dtype=mean.dtype,generator=generator).unsqueeze(-1)).squeeze(-1);paths=[];dynamics=base.dynamics;cache={};last=float(last_time)
 linear=hasattr(dynamics,'F') and not hasattr(dynamics,'f')
 if not linear:assert dynamics.fixed_diffusion,'Only author default constant diagonal diffusion is qualified here'
 for target in future_times:
  target=float(target);delta=target-last;assert delta>0
  if linear:
   key=round(delta,10)
   if key not in cache:
    zero=state.new_zeros(1,base.z_dim);zero_cov=state.new_zeros(1,base.z_dim,base.z_dim);_,noise_chol,_=base.predict_step(zero,zero_cov,dynamics,0.,delta,step_size,'matrix_exp');cache[key]=(torch.matrix_exp(dynamics.F*delta),noise_chol[0])
   A,L=cache[key];noise=torch.randn(state.shape,device=state.device,dtype=state.dtype,generator=generator);state=state@A.T+noise@L.T
  else:
   steps=math.ceil(delta/step_size);h=delta/steps
   for _ in range(steps):
    diffusion=dynamics.GQGt(state).diagonal(dim1=-2,dim2=-1).sqrt();noise=torch.randn(state.shape,device=state.device,dtype=state.dtype,generator=generator);state=state+h*dynamics.f(state)+math.sqrt(h)*diffusion*noise
  paths.append(state);last=target
 return torch.stack(paths,1)

@torch.no_grad()
def sample_joint(model,history,mask,past_times,future_times,n,draw_seed,step_size=.0125,return_states=False):
 device=next(model.parameters()).device;dtype=next(model.parameters()).dtype;history=torch.as_tensor(history,device=device,dtype=dtype);mask=torch.as_tensor(mask,device=device,dtype=dtype);past_times=torch.as_tensor(past_times,device=device,dtype=dtype);future_times=torch.as_tensor(future_times,device=device,dtype=dtype);B=len(history);hist=history.repeat_interleave(n,0);masks=mask.repeat_interleave(n,0);devices=[device.index or 0] if device.type=='cuda' else []
 with torch.random.fork_rng(devices=devices):
  torch.manual_seed(draw_seed);aux,_,_=model.aux_inference_net(hist,masks,num_samples=1,deterministic=False);filtered=model.base_ssm.filter(aux[0],masks,past_times);mean,chol=filtered['last_filtered_dist'];generator=torch.Generator(device=device).manual_seed(draw_seed+1);states=state_paths(model.base_ssm,mean,chol,past_times[-1],future_times,generator,step_size);torch.manual_seed(draw_seed+2);outputs=[]
  for t in range(len(future_times)):
   auxiliary=model.base_ssm.emit(states[:,t]);outputs.append(model.y_emission_net(auxiliary).sample())
  values=torch.stack(outputs,1).reshape(B,n,len(future_times),-1)
 if return_states:return values,states.reshape(B,n,len(future_times),-1)
 return values
