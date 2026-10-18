"""Frozen C81 development screen. No reserved evaluation trajectories are scored."""
import sys,pathlib,time,itertools
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common81_v1 import *
from features81_v1 import NativeFeatures,load_assets,qualify,old
import ordinary80_v1 as ordinary
import numpy as np
import torch
from ode_tutorial_helper import load_odebench_system
torch.set_num_threads(2);torch.set_num_interop_threads(2)
KINDS=['fim_readout_residual','fim_readout_zero','rff_residual','affine_residual']

def weak(nf,x,t,kind,ridge):
 phi,base=nf.phi(x,kind);z=(x-nf.m)/nf.s;step=np.diff(t)[:,None];ip=np.r_[np.zeros((1,phi.shape[1])),np.cumsum(.5*(phi[1:]+phi[:-1])*step,axis=0)];ib=np.r_[np.zeros((1,2)),np.cumsum(.5*(base[1:]+base[:-1])/nf.s*step,axis=0)];A=[];B=[]
 for w in [8,16,32]:
  A.append(ip[w:]-ip[:-w]);b=z[w:]-z[:-w]
  if kind!='fim_readout_zero':b=b-(ib[w:]-ib[:-w])
  B.append(b)
 A=np.concatenate(A);B=np.concatenate(B);scale=np.maximum(np.sqrt(np.mean(A*A,axis=0)),1e-10);Z=A/scale;coefficient=np.linalg.solve(Z.T@Z+ridge*np.eye(Z.shape[1]),Z.T@B);coef=coefficient/scale[:,None]
 model=dict(kind=kind,ridge=float(ridge),coefficient=coef.tolist(),mean=nf.m.tolist(),std=nf.s.tolist(),weak_column_rms=scale.tolist(),rows=len(A),features=phi.shape[1],correction_parameters=2*phi.shape[1],weak_fit_mse=float(np.mean((A@coef-B)**2)),context_steps=len(nf.x))
 def fn(xq):
  h,f=nf.phi(xq,kind);return ((f if kind!='fim_readout_zero' else 0)+h@coef*nf.s)[0]
 return fn,model

def main():
 started=time.perf_counter();cohort=read(OUT/'COHORT81_V1.json');assert cohort['protocol_sha256']==sha(ROOT/'PROTOCOL81_V1.md');assert not (OUT/'OPENING81_V1.json').exists();source_count=old.check_source();model,weights,data=load_assets();art=storage('artifact')/'screen_v1';art.mkdir(exist_ok=False)
 qualification=qualify(model,data,cohort['development'][0]);dump(OUT/'QUALIFICATION81_V1.json',dict(utc=utc(),status='PASS_CONTEXT_ONLY_READOUT_QUALIFICATION',system=cohort['development'][0],checks=qualification,weights_sha256=weights['sha256'],source_files_checked=source_count))
 dump(OUT/'OPENING81_V1.json',dict(utc=utc(),protocol_sha256=sha(ROOT/'PROTOCOL81_V1.md'),cohort_sha256=sha(OUT/'COHORT81_V1.json'),qualification_sha256=sha(OUT/'QUALIFICATION81_V1.json'),development=cohort['development'],evaluation_opened=False,device='cpu',affinity=ordinary.AFFINITY,new_neural_training_fits=0,model_parameters=sum(p.numel() for p in model.parameters())))
 rows=[];candidates=[];models=[]
 for sid in cohort['development']:
  state('PROGRESS81_V1.json',dict(status='DEVELOPMENT_RUNNING',system=sid,closed_rows=len(rows),candidate_fits=len(candidates),elapsed_seconds=time.perf_counter()-started))
  tt,xx,_=load_odebench_system(sid,data['path'],noise_sigma=.03,subsample_fraction=.5,rng=np.random.default_rng(2026810));x=xx[0].copy();t=tt[0,:,0].copy();assert x.shape==(256,2);fitstart=time.perf_counter();prefix=NativeFeatures(model,x[:192],t[:192]);full=NativeFeatures(model,x,t);context_encoding_seconds=time.perf_counter()-fitstart;selected={'native':(lambda y:full.base(y)[0],dict(kind='native',features=full.p,correction_parameters=0,adaptation_seconds=context_encoding_seconds))}
  for kind in KINDS:
   fitstart=time.perf_counter();local=[]
   for ridge in [.001,.1,10.]:
    fn,parameters=weak(prefix,x[:192],t[:192],kind,ridge);p,solver=ordinary.integrate(fn,x[191],t[191:]);score=ordinary.metric(p,x[191:]);row=dict(system=sid,kind=kind,ridge=ridge,validation=score,solver=solver,parameters=parameters);local.append(row);candidates.append(row)
   valid=[r for r in local if r['solver']['success'] and r['validation']['finite_complete']]
   if not valid:selected[kind]=(None,dict(kind=kind,all_context_candidates_failed=True,adaptation_seconds=time.perf_counter()-fitstart));continue
   chosen=min(valid,key=lambda r:(r['validation']['mse'],r['ridge']));fn,parameters=weak(full,x,t,kind,chosen['ridge']);parameters['selected_context_validation_mse']=chosen['validation']['mse'];parameters['adaptation_seconds']=time.perf_counter()-fitstart+context_encoding_seconds;selected[kind]=(fn,parameters)
  local=[];fitstart=time.perf_counter()
  for library,ridge,threshold in itertools.product(['polynomial3','polynomial3_trig'],[1e-6,.001,1.],[0.,.03]):
   pars=ordinary.fit(t[:192],x[:192],library,ridge,threshold);p,solver=ordinary.integrate(ordinary.field(pars),x[191],t[191:]);score=ordinary.metric(p,x[191:]);row=dict(system=sid,kind='ordinary_weak_dictionary',library=library,ridge=ridge,threshold=threshold,validation=score,solver=solver,parameters=pars);local.append(row);candidates.append(row)
  valid=[r for r in local if r['solver']['success'] and r['validation']['finite_complete']]
  if valid:
   chosen=min(valid,key=lambda r:(r['validation']['mse'],r['library'],r['ridge'],r['threshold']));pars=ordinary.fit(t,x,chosen['library'],chosen['ridge'],chosen['threshold']);pars['adaptation_seconds']=time.perf_counter()-fitstart;pars['correction_parameters']=2*len(pars['coefficient']);selected['ordinary_weak_dictionary']=(ordinary.field(pars),pars)
  else:selected['ordinary_weak_dictionary']=(None,dict(all_context_candidates_failed=True))
  # Only now load clean scoring targets; no target is given to fitting/selection.
  clean_t,clean,_=load_odebench_system(sid,data['path'],noise_sigma=0.,subsample_fraction=1.,rng=np.random.default_rng(2026810))
  for kind,(fn,pars) in selected.items():
   modelpath=art/f'system{sid}_{kind}_model.json';dump(modelpath,pars);models.append(dict(system=sid,kind=kind,path=str(modelpath),sha256=sha(modelpath)))
   for k in [0,1]:
    y=clean[k];times=clean_t[k,:,0]
    if fn is None:p=None;solver=dict(success=False,message='All inner-context candidates failed',nfev=0,wall_seconds=0)
    else:p,solver=ordinary.integrate(fn,y[0],times)
    path=art/f'system{sid}_ic{k}_{kind}.npz';np.savez_compressed(path,prediction=p if p is not None else np.empty((0,2)),target=y,time=times,context=x,context_times=t);row=dict(system=sid,initial_condition_index=k,role='context_reconstruction' if k==0 else 'independent_new_initial_condition',kind=kind,**ordinary.metric(p,y),solver=solver,parameters=pars.get('correction_parameters'),selected_ridge=pars.get('ridge'),adaptation_seconds=pars.get('adaptation_seconds'),array_path=str(path),array_sha256=sha(path));rows.append(row);print(json.dumps({key:row[key] for key in ['system','initial_condition_index','kind','nrmse','solver']}),flush=True)
  state('PROGRESS81_V1.json',dict(status='DEVELOPMENT_RUNNING',system=sid,closed_rows=len(rows),candidate_fits=len(candidates),elapsed_seconds=time.perf_counter()-started))
 assert len(rows)==72 and len(candidates)==144 and len(models)==36;assert old.check_source()==source_count and sha(weights['path'])==weights['sha256'];assert sha(ROOT/'PROTOCOL81_V1.md')==cohort['protocol_sha256']
 dump(OUT/'RESULTS81_V1.json',dict(utc=utc(),status='SIX_SYSTEM_DEVELOPMENT_SCREEN_CLOSED_UNVERIFIED',rows=rows,candidates=candidates,models=models,evaluation_opened=False,context_noise_seed=2026810,new_neural_training_fits=0,new_linear_candidate_fits=len(candidates),full_context_models=len(models),weights_unchanged=True,source_files_checked=source_count,wall_seconds=time.perf_counter()-started,accepted_final_directions=0,overall_goal='ACTIVE_UNACHIEVED'))
 state('PROGRESS81_V1.json',dict(status='SIX_SYSTEM_DEVELOPMENT_SCREEN_CLOSED_UNVERIFIED',closed_rows=len(rows),wall_seconds=time.perf_counter()-started))
if __name__=='__main__':main()
