"""Exactly the three fixed native tutorial systems; distinguish reconstruction from new IC."""
import sys,pathlib,time,copy,importlib.metadata
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from native80_v1 import *
sys.path.insert(0,str(PROJECT/'cycle71/scripts'))
from runtime_affinity71_v1 import configure
AFFINITY=configure('efficient')
import numpy as np
import torch
torch.set_num_threads(2);torch.set_num_interop_threads(2)
from fim.models.ode import load_fim_ode_local
from safetensors.torch import load_file
from ode_tutorial_helper import load_odebench_system,make_fim_vf_fn,integrate_from_context
import scipy.integrate
os.environ['MPLBACKEND']='Agg'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def main():
 started=time.perf_counter();assert read(OUT/'IMPORT_AUDIT80_V3.json')['status']=='PASS_NATIVE_FULL_PACKAGE_IMPORT'
 sourcecount=check_source();a=assets();weights=next(x for x in a if x['relative'].endswith('.safetensors'));config=next(x for x in a if x['relative'].endswith('config.json'));data=next(x for x in a if x['relative'].endswith('strogatz_extended.json'))
 model=load_fim_ode_local(pathlib.Path(weights['path']).parent,device='cpu');model.eval();stored={k.removeprefix('model.'):t for k,t in load_file(weights['path'],device='cpu').items()};actual=model.state_dict();assert set(actual)==set(stored)
 for name in actual:assert torch.equal(actual[name],stored[name]),name
 parameters=sum(p.numel() for p in model.parameters());assert parameters==12968068,parameters
 art=storage('artifact')/'native_tutorial_v2';art.mkdir(exist_ok=False);versions={name:importlib.metadata.version(name) for name in ['torch','numpy','scipy','matplotlib','transformers','huggingface_hub','safetensors','peft','optree']}
 dump(OUT/'OPENING80_V2.json',dict(utc=utc(),protocol_sha256=sha(ROOT/'PROTOCOL80_V1.md'),source_manifest_sha256=sha(OUT/'SOURCE_MANIFEST80_V1.json'),asset_manifest_sha256=sha(OUT/'ASSET_MANIFEST80_V1.json'),parameters=parameters,exact_state_tensors_checked=len(actual),versions=versions,affinity=AFFINITY,device='cpu',new_training_fits=0,context_seed=0,initial_conditions=[0,1],native_source_modified=False))
 rows=[];checks=[];fig,axes=plt.subplots(3,2,figsize=(10,12),layout='constrained')
 for i,sid in enumerate([26,42,28]):
  state('PROGRESS80_V2.json',dict(status='NATIVE_TUTORIAL_RUNNING',system=sid,closed_rows=len(rows)))
  times,noisy,meta=load_odebench_system(sid,data['path'],noise_sigma=.03,subsample_fraction=.5,rng=np.random.default_rng(0));clean_times,truth,cleanmeta=load_odebench_system(sid,data['path'],noise_sigma=0.,subsample_fraction=1.,rng=np.random.default_rng(0));assert len(noisy)>=2 and noisy.shape[-1]==2
  context=noisy[:1].copy();ct=times[:1].copy();query=np.concatenate([context[0,np.linspace(0,len(context[0])-1,16,dtype=int)],truth[:2,0]],axis=0);cached=make_fim_vf_fn(model,context,ct,'cpu');cached_pred=cached(query);repeat=cached(query);assert np.array_equal(cached_pred,repeat)
  trajectories=torch.tensor(context,dtype=torch.float32).unsqueeze(0);tt=torch.tensor(ct,dtype=torch.float32).unsqueeze(0);mask=torch.ones((*trajectories.shape[:-1],1),dtype=torch.bool)
  with torch.no_grad():direct=model.get_prediction_for_eval(model.model_forward(trajectories,tt,torch.tensor(query,dtype=torch.float32).unsqueeze(0),mask))[0,:,:2].numpy()
  difference=float(abs(direct-cached_pred).max());assert np.allclose(direct,cached_pred,atol=2e-5,rtol=2e-5)
  check=dict(system=sid,direct_vs_native_cached_max_abs=difference,repeated_native_cached_max_abs=float(abs(repeat-cached_pred).max()),query_count=len(query),context_paths=len(context),context_steps=context.shape[1],available_clean_paths=len(truth),meta=meta);checks.append(check)
  for k in [0,1]:
   actual_calls=[];solve_original=scipy.integrate.solve_ivp
   def recorded_solve(*args,**kwargs):
    start=time.perf_counter();sol=solve_original(*args,**kwargs);actual_calls.append(dict(success=bool(sol.success),message=str(sol.message),status=int(sol.status),nfev=int(sol.nfev),njev=int(sol.njev),nlu=int(sol.nlu),returned_steps=int(len(sol.t)),wall_seconds=time.perf_counter()-start,method=kwargs.get('method'),rtol=kwargs.get('rtol'),atol=kwargs.get('atol')));return sol
   scipy.integrate.solve_ivp=recorded_solve;exception=None;prediction=None
   try:prediction=integrate_from_context(model,context,ct[0,:,0],truth[k,0].copy(),clean_times[k,:,0].copy(),'cpu')
   except Exception as exc:exception=repr(exc)
   finally:scipy.integrate.solve_ivp=solve_original
   target=truth[k].copy();finite=prediction is not None and prediction.shape==target.shape and np.isfinite(prediction).all();err=prediction-target if finite else None;mse=float(np.mean(err**2)) if finite else None;mae=float(np.mean(abs(err))) if finite else None;std=float(np.std(target));role='context_trajectory_reconstruction' if k==0 else 'new_initial_condition_same_system'
   path=art/f'system{sid}_ic{k}.npz';np.savez_compressed(path,context=context,context_times=ct,query=query,cached_field=cached_pred,direct_field=direct,initial_state=target[0],time=clean_times[k,:,0],prediction=prediction if prediction is not None else np.empty((0,2)),target=target)
   row=dict(system=sid,initial_condition_index=k,role=role,context_contains_this_trajectory=k==0,finite_complete=bool(finite),mse=mse,rmse=float(np.sqrt(mse)) if finite else None,mae=mae,target_std_evaluation_only=std,nrmse=float(np.sqrt(mse)/std) if finite and std else None,solver_calls=actual_calls,exception=exception,array_path=str(path),array_sha256=sha(path));rows.append(row)
   ax=axes[i,k];ax.plot(target[:,0],target[:,1],color='black',linestyle='--',label='clean reference');ax.scatter(context[0,:,0],context[0,:,1],s=4,color='gray',alpha=.4,label='noisy context')
   if prediction is not None and prediction.ndim==2 and len(prediction):ax.plot(prediction[:,0],prediction[:,1],color='#0072B2',label='native FIM-ODE')
   ax.set_title(f'ODEBench {sid}, IC {k}: '+('reconstruction' if k==0 else 'new IC'));ax.set_xlabel('state 0');ax.set_ylabel('state 1');ax.legend(fontsize=7)
   state('PROGRESS80_V2.json',dict(status='NATIVE_TUTORIAL_RUNNING',system=sid,initial_condition=k,closed_rows=len(rows),elapsed_seconds=time.perf_counter()-started));print(json.dumps({key:row[key] for key in ['system','initial_condition_index','role','finite_complete','mse','nrmse','solver_calls','exception']}),flush=True)
 figure=art/'native_phase_portraits80_v2.png';fig.savefig(figure,dpi=160);plt.close(fig);assert len(rows)==6 and check_source()==sourcecount and sha(weights['path'])==weights['sha256']
 dump(OUT/'NATIVE_RESULTS80_V2.json',dict(utc=utc(),status='SIX_FIXED_NATIVE_TUTORIAL_INITIAL_CONDITIONS_CLOSED_UNVERIFIED',source_files_checked=sourcecount,rows=rows,qualification_checks=checks,parameters=parameters,weights_sha256=weights['sha256'],data_sha256=data['sha256'],figure_path=str(figure),figure_sha256=sha(figure),new_training_fits=0,full_paper_reproduction=False,independent_general_mechanism_confirmation=False,accepted_final_directions=0,overall_goal='ACTIVE_UNACHIEVED',wall_seconds=time.perf_counter()-started))
 state('PROGRESS80_V2.json',dict(status='SIX_FIXED_NATIVE_TUTORIAL_INITIAL_CONDITIONS_CLOSED_UNVERIFIED',closed_rows=len(rows),wall_seconds=time.perf_counter()-started))
if __name__=='__main__':main()
