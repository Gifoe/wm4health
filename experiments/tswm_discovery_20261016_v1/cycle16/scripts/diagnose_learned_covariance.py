"""Read-only full-model covariance gate; no new-method or posterior-truth claim."""
import runtime_profile
import datetime,json,pathlib,sys,time
import numpy as np
import torch
from config_tools import load
from data_protocol import Data,L,H,T,metrics,sha
ROOT=pathlib.Path(__file__).resolve().parents[1];SOURCE=ROOT.parent/'cycle15/upstream/MTS3'
sys.path.insert(0,str(SOURCE))
from agent.worldModels.MTS3 import MTS3

def full(packed):
 u,l,s=packed.chunk(3,dim=-1)
 return torch.cat([torch.cat([torch.diag_embed(u),torch.diag_embed(s)],dim=-1),
                   torch.cat([torch.diag_embed(s),torch.diag_embed(l)],dim=-1)],dim=-2)

def packed(matrix):
 d=matrix.shape[-1]//2
 return torch.cat([matrix[...,:d,:d].diagonal(dim1=-2,dim2=-1),matrix[...,d:,d:].diagonal(dim1=-2,dim2=-1),
                   matrix[...,:d,d:].diagonal(dim1=-2,dim2=-1)],dim=-1)

def blocks(parts):
 a,b,c,d=[x.detach().double() for x in parts]
 return torch.cat([torch.cat([a,b],dim=-1),torch.cat([c,d],dim=-1)],dim=-2)

def recurrences(A,C,Q,P0,S,horizon):
 Pn=P0.clone();Pi=P0.clone();Pj=P0.clone();R=torch.zeros_like(P0)
 n=[];i=[];j=[];r=[];U=C@S@C.T
 for h in range(horizon):
  Pn=full(packed(A@Pn@A.T+U+Q))
  Pi=A@Pi@A.T+U+Q
  Pj=A@Pj@A.T+U+Q+A@R@C.T+C@R.transpose(-1,-2)@A.T
  R=A@R+C@S
  n.append(packed(Pn));i.append(packed(Pi));j.append(packed(Pj));r.append(torch.linalg.matrix_norm(R))
 return torch.stack(n,1),torch.stack(i,1),torch.stack(j,1),torch.stack(r,1)

def analytical_checks():
 # Independent closed-form scalar variance of x_h=a^h*x0+c*sum(a^i)*z+sum(a^i)*eps.
 worst=0.
 for a in [0.,.5,.95,1.,-.8]:
  for c in [0.,.4,1.]:
   A=torch.tensor([[a]],dtype=torch.float64);C=torch.tensor([[c]],dtype=torch.float64)
   P=torch.tensor([[[.7]]],dtype=torch.float64);S=torch.tensor([[[.3]]],dtype=torch.float64);Q=torch.tensor([[.05]],dtype=torch.float64)
   joint=P.clone();R=torch.zeros_like(P)
   for h in range(1,76):
    joint=A@joint@A.T+C@S@C.T+Q+A@R@C.T+C@R.transpose(-1,-2)@A.T;R=A@R+C@S
    exact=a**(2*h)*.7+c*c*.3*sum(a**k for k in range(h))**2+.05*sum(a**(2*k) for k in range(h))
    worst=max(worst,abs(float(joint.item())-exact));assert np.isclose(joint.item(),exact,rtol=1e-11,atol=1e-11)
 return worst

def main(fixed_step=None,max_origins=None,smoke=False):
 start=time.perf_counter();torch.set_num_threads(1)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 checks={'analytical_max_absolute_error':analytical_checks()}
 train=ROOT/'outputs/train/NP_MTS3_seed2021'
 if fixed_step is None:
  fit=json.loads((train/'results.json').read_text());assert fit['training_complete']
  pointer=json.loads((train/'best_checkpoint.json').read_text());selection='Completed fit best validation NLL'
 else:
  assert smoke and fixed_step==100 and max_origins==8
  path=train/f'checkpoint_step{fixed_step:06d}.pt'
  pointer={'step':fixed_step,'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path)};selection='Fixed step100 smoke test only'
 step=pointer['step'];out=ROOT/f'outputs/diagnostics/{"smoke" if smoke else "final"}_step{step:06d}'
 assert not out.exists();out.mkdir(parents=True)
 checkpoint=ROOT/pointer['path'];assert sha(checkpoint)==pointer['sha256']
 manifest=json.loads((ROOT.parent/'cycle15/outputs/source/manifest.json').read_text())
 for row in manifest['files']:assert sha(SOURCE/row['path'])==row['sha256']
 payload=torch.load(checkpoint,map_location='cuda',weights_only=True)
 assert payload['step']==step and payload['config_sha256']==sha(train/'config.json')
 data=Data();roots=data.val if max_origins is None else data.val[:max_origins]
 model=MTS3(input_shape=(1,),action_dim=6,config=load(SOURCE/'experiments/mobileRobot/conf/model/default_mts3.yaml')).cuda()
 model.load_state_dict(payload['model'],strict=True);model.eval();del payload
 A=blocks(model._state_predict._A);C=blocks(model._state_predict._C)
 Q=full(torch.cat([v.detach().double() for v in model._state_predict.get_process_noise()],dim=-1))[0]
 assert torch.allclose(Q,torch.diag(Q.diagonal()))
 store={k:[] for k in ['target','native_mean','native_variance','dense_independent_variance','dense_joint_variance',
                       'native_latent_covariance_summary','dense_independent_latent_covariance_summary','dense_joint_latent_covariance_summary',
                       'context_covariance_summary','initial_fast_covariance_summary','context_to_process_trace_ratio','joint_crosscov_frobenius']}
 reproduce_error=[];context_errors=[];no_update_errors=[];mean_decode_errors=[];h1_errors=[]
 with torch.inference_mode():
  for lo in range(0,len(roots),8):
   obs,cond,mask,target=[torch.as_tensor(v,device='cuda') for v in data.batch(roots[lo:lo+8],'val')]
   calls=[];decoder=[]
   def capture(module,args,output):
    means,covs=args
    calls.append({'fast_cov':torch.cat(covs[0],dim=-1).detach().clone(),
                  'context_cov':torch.cat(covs[1],dim=-1).detach().clone(),
                  'context_mean':means[-1].detach().clone(),
                  'next_cov':torch.cat(output[1],dim=-1).detach().clone()})
   def capture_decoder(module,args):decoder.append(tuple(v.detach().clone() for v in args))
   hook=model._state_predict.register_forward_hook(capture);dhook=model._obsDec.register_forward_pre_hook(capture_decoder)
   try:mean,var=model(obs,cond,mask)[:2]
   finally:hook.remove();dhook.remove()
   assert len(calls)==T and len(decoder)==1
   calls=calls[L-1:T];native_mean=mean[:,L-1:T];native_var=var[:,L-1:T]
   ctx=calls[0]['context_cov'];ctxmean=calls[0]['context_mean']
   context_errors.append(max(max(float((x['context_cov']-ctx).abs().max()),float((x['context_mean']-ctxmean).abs().max())) for x in calls))
   no_update_errors.append(max(float((calls[h]['fast_cov']-calls[h-1]['next_cov']).abs().max()) for h in range(1,H)))
   assert context_errors[-1]==no_update_errors[-1]==0
   P0=full(calls[0]['fast_cov'].double());S=full(ctx.double())
   native_cov=torch.stack([x['next_cov'] for x in calls],1)
   replay,independent,joint,Rnorm=recurrences(A,C,Q,P0,S,H)
   error=float((replay-native_cov.double()).abs().max());reproduce_error.append(error)
   assert torch.allclose(replay,native_cov.double(),rtol=2e-5,atol=2e-5),(error,step)
   h1_errors.append(float((joint[:,0]-independent[:,0]).abs().max()));assert h1_errors[-1]==0
   latent_mean,latent_cov=decoder[0];latent_mean=latent_mean[:,L-1:T]
   assert torch.equal(latent_cov[:,L-1:T],native_cov)
   decoded_native=model._obsDec(latent_mean,native_cov)
   assert torch.allclose(decoded_native[0],native_mean,rtol=1e-6,atol=1e-6) and torch.allclose(decoded_native[1],native_var,rtol=1e-6,atol=1e-6)
   other=[]
   for cov in [independent,joint]:
    m,v=model._obsDec(latent_mean,cov.float());mean_decode_errors.append(float((m-native_mean).abs().max()))
    assert torch.allclose(m,native_mean,rtol=1e-6,atol=1e-6) and torch.isfinite(v).all() and (v>0).all();other.append(v)
   ratio=(C@S@C.T).diagonal(dim1=-2,dim2=-1).sum(-1)/Q.diagonal().sum()
   values=[target,native_mean,native_var,*other,native_cov,independent,joint,ctx,calls[0]['fast_cov'],ratio,Rnorm]
   for key,value in zip(store,values):store[key].append(value.cpu().numpy())
   print(json.dumps({'diagnostic_step':step,'smoke':smoke,'origins_completed':lo+len(obs),'native_replay_max_error':error}),flush=True)
 arrays={key:np.concatenate(value) for key,value in store.items()};arrays.update(origins=roots,A=A.cpu().numpy(),C=C.cpu().numpy(),Q=Q.cpu().numpy())
 np.savez_compressed(out/'arrays.npz',**arrays)
 rows=[]
 for name in ['native','dense_independent','dense_joint']:
  v=arrays[name+'_variance'];original=arrays['native_variance']
  rows.append({'control':name,**metrics(arrays['native_mean'],v,arrays['target']),
               'variance_relative_change_mean':float(np.mean(np.abs(v-original)/original)),
               'variance_relative_change_max':float(np.max(np.abs(v-original)/original)),
               'horizon_mean_predicted_variance':v.mean(axis=(0,2)).tolist()})
 native=arrays['native_latent_covariance_summary'];independent=arrays['dense_independent_latent_covariance_summary'];joint=arrays['dense_joint_latent_covariance_summary']
 d=A.shape[-1]//2
 trace=lambda x:x[:,:,:2*d].sum(axis=-1)
 checks.update(native_replay_max_absolute_error=max(reproduce_error),context_constancy_error=max(context_errors),
               future_no_update_error=max(no_update_errors),decoded_mean_error=max(mean_decode_errors),dense_horizon1_error=max(h1_errors))
 result={'checkpoint':pointer,'selection':selection,'smoke_only':smoke,'validation_origins':len(roots),'rows':rows,'checks':checks,
         'source_revision':manifest['revision'],'data_registry_sha256':sha(ROOT/'outputs/data/registry.json'),
         'script_sha256':sha(pathlib.Path(__file__)),'plan_sha256':sha(ROOT/'DIAGNOSTIC_PLAN.md'),
         'dependency_sha256':{name:sha(ROOT/'scripts'/name) for name in ['data_protocol.py','config_tools.py','runtime_profile.py']},
         'context_to_process_trace_ratio_quantiles':np.quantile(arrays['context_to_process_trace_ratio'],[0,.25,.5,.75,1]).tolist(),
         'independent_to_native_trace_ratio_by_horizon':np.mean(trace(independent)/trace(native),axis=0).tolist(),
         'joint_to_independent_trace_ratio_by_horizon':np.mean(trace(joint)/trace(independent),axis=0).tolist(),
         'assumptions':['Same factorized initial fast/context belief; initial cross-covariance is set to zero, not inferred.',
                        'Same native means and unchanged native decoder; changed covariance inputs can be out of decoder training distribution.',
                        'Full within-fast-state covariance control separates spatial projection from persistent-context dependence.',
                        'Validation-only descriptive inference diagnostic, not proof of true posterior or a trained correction.'],
         'arrays_sha256':sha(out/'arrays.npz'),'elapsed_seconds':time.perf_counter()-start,
         'no_test_evaluation':True,'new_method_claim':False,'objective_achieved':False,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 (out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
 print(json.dumps({k:v for k,v in result.items() if k not in ['rows','independent_to_native_trace_ratio_by_horizon','joint_to_independent_trace_ratio_by_horizon']}),flush=True)
 print(json.dumps({'rows':[{k:v for k,v in row.items() if k!='horizon_mean_predicted_variance'} for row in rows]}),flush=True)

if __name__=='__main__':main()
