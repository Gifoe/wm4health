from common88_v1 import *
import numpy as np,yaml,time
import torch
torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
M=read(O/'SOURCE_MANIFEST88_V1.json');SOURCE=Path(M['source_root']);sys.path.insert(0,str(SOURCE))
from agent.worldModels.MTS3 import MTS3
class AttrDict(dict):
 def __getattr__(self,key):
  try:return self[key]
  except KeyError:raise AttributeError(key) from None
def attrs(x):
 if isinstance(x,dict):return AttrDict({k:attrs(v) for k,v in x.items()})
 if isinstance(x,list):return [attrs(v) for v in x]
 return x
def config():
 p=SOURCE/'experiments/mobileRobot/conf/model/default_mts3.yaml';c=yaml.safe_load(p.read_text(encoding='utf-8'));c['wandb']['log']=False;return attrs(c)
def main():
 started=time.perf_counter()
 for row in M['files']:assert sha(SOURCE/row['path'])==row['sha256']
 torch.manual_seed(2026880);rng=np.random.default_rng(2026880);device='cuda' if torch.cuda.is_available() else 'cpu';cfg=config();H=cfg.mts3.time_scale_multiplier;T=2*H;B=2
 # A numerical smoke fixture only, not the proposed nonlinear scientific test.
 action=rng.normal(0,.2,(B,T,1)).astype(np.float32);state=np.zeros((B,T+1,1),np.float32)
 for t in range(T):state[:,t+1]=.8*state[:,t]+.1*action[:,t]+rng.normal(0,.01,(B,1))
 obs=state[:,:T];target=state[:,1:];valid=np.ones((B,T,1),bool);valid[:,H:]=False
 x=torch.as_tensor(obs,device=device);a=torch.as_tensor(action,device=device);mask=torch.as_tensor(valid,device=device);y=torch.as_tensor(target,device=device)
 model=MTS3(input_shape=(1,),action_dim=1,config=cfg,use_cuda_if_available=device=='cuda').to(device);params=sum(p.numel() for p in model.parameters());opt=torch.optim.Adam(model.parameters(),lr=.003);trace=[];old=next(model.parameters()).detach().clone()
 for step in range(3):
  model.train();opt.zero_grad(set_to_none=True);out=model(x,a,mask);mean,var=out[0],out[1];assert mean.shape==y.shape and var.shape==y.shape;assert torch.isfinite(mean).all() and torch.isfinite(var).all() and (var>0).all();loss=.5*(torch.log(2*torch.pi*var)+(mean-y)**2/var).mean();loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(loss) and torch.isfinite(norm);grad_count=sum(p.grad is not None for p in model.parameters());assert grad_count>0;opt.step();trace.append(dict(step=step,nll=float(loss.detach()),grad_norm=float(norm),parameters_with_gradient=grad_count));log(dict(stage='native_mts3_gradient_smoke',**trace[-1]))
 assert not torch.equal(old,next(model.parameters()).detach());model.eval()
 with torch.no_grad():
  base=model(x,a,mask);changed=x.clone();changed[:,H:]+=1.;masked=model(changed,a,mask);masked_delta=float(abs(base[0]-masked[0]).max());masked_var_delta=float(abs(base[1]-masked[1]).max());assert masked_delta<1e-6 and masked_var_delta<1e-6
  tail_start=T-10;tail=a.clone();tail[:,tail_start:]+=.4;tail_result=model(x,tail,mask);early=slice(H-1,tail_start-1);tail_delta=float(abs(base[0][:,early]-tail_result[0][:,early]).max());first_delta=float(abs(base[0][:,H-1]-tail_result[0][:,H-1]).max());tail_var_delta=float(abs(base[1][:,early]-tail_result[1][:,early]).max())
 raw=Path(STORAGE['artifact_root'])/'cycle88';raw.mkdir(parents=True,exist_ok=True);fixture=raw/'native_smoke_fixture88_v1.npz'
 with fixture.open('xb') as f:np.savez_compressed(f,observations=obs,actions=action,target=target,mask=valid,base_mean=base[0].cpu().numpy(),base_variance=base[1].cpu().numpy(),masked_mean=masked[0].cpu().numpy(),tail_mean=tail_result[0].cpu().numpy(),tail_variance=tail_result[1].cpu().numpy())
 cp=Path(STORAGE['checkpoint_root'])/'cycle88';cp.mkdir(parents=True,exist_ok=True);weights=cp/'native_mts_smoke88_v1.pt';torch.save(model.state_dict(),weights)
 receipt=dict(utc=utc(),status='NATIVE_RUNTIME_GRADIENT_AND_MASK_QUALIFIED_NOT_SCIENTIFIC_FAILURE',source_commit=M['commit'],source_files_verified=len(M['files']),source_code_modified=False,configuration='unchanged official mobileRobot model dimensions/H; input dimension1/action dimension1; logging disabled; plain attribute dictionary replaces Hydra config loading only',parameters=params,H=H,T=T,batch=B,device=device,torch_version=torch.__version__,matmul_precision=torch.get_float32_matmul_precision(),allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),trace=trace,masked_future_observation_mean_max_abs=masked_delta,masked_future_observation_variance_max_abs=masked_var_delta,late_action_start=tail_start,earlier_forecast_mean_tail_sensitivity=tail_delta,first_forecast_step_tail_sensitivity=first_delta,earlier_forecast_variance_tail_sensitivity=tail_var_delta,tail_sensitivity_scope='three-step smoke weights only; architectural potential, NOT evidence about trained paper model or new research gap',output_shapes=[list(z.shape) for z in base],fixture_path=str(fixture),fixture_sha256=sha(fixture),weights_path=str(weights),weights_sha256=sha(weights),source_manifest_sha256=sha(O/'SOURCE_MANIFEST88_V1.json'),new_scientific_training_fits=0,elapsed_seconds=time.perf_counter()-started)
 write(O/'NATIVE_QUALIFICATION88_V1.json',receipt);log(dict(stage='native_mts3_qualification_closed',parameters=params,masked_future_obs_delta=masked_delta,earlier_forecast_tail_sensitivity=tail_delta,first_forecast_tail_sensitivity=first_delta,scientific_fits=0))
if __name__=='__main__':main()
