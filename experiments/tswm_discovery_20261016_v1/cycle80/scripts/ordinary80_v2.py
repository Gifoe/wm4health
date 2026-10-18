"""Standard weak-integral vector-field regression using only one available noisy context."""
import sys,pathlib,time,itertools
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from native80_v1 import *
sys.path.insert(0,str(PROJECT/'cycle71/scripts'))
from runtime_affinity71_v1 import configure
AFFINITY=configure('efficient')
import numpy as np
from scipy.integrate import solve_ivp
import sympy as sp
POWERS=[(i,j) for degree in range(4) for i in range(degree+1) for j in [degree-i]]
def features(x,m,s,library):
 z=(np.asarray(x)-m)/s;columns=[z[...,0]**i*z[...,1]**j for i,j in POWERS]
 if library=='polynomial3_trig':columns += [np.sin(x[...,j]) for j in [0,1]]+[np.cos(x[...,j]) for j in [0,1]]
 return np.stack(columns,axis=-1)
def fit(t,x,library,ridge,threshold):
 m=x.mean(0);s=x.std(0);s=np.maximum(s,1e-8);theta=features(x,m,s,library);step=.5*(theta[1:]+theta[:-1])*np.diff(t)[:,None];integral=np.concatenate([np.zeros((1,theta.shape[1])),np.cumsum(step,axis=0)]);z=(x-m)/s;aa=[];bb=[]
 for width in [8,16,32]:aa.append(integral[width:]-integral[:-width]);bb.append(z[width:]-z[:-width])
 A=np.concatenate(aa);B=np.concatenate(bb);scale=np.maximum(np.sqrt(np.mean(A*A,axis=0)),1e-10);Z=A/scale;coef=np.zeros((Z.shape[1],2));active=np.ones_like(coef,dtype=bool)
 for iteration in range(5):
  for j in range(2):
   ids=active[:,j]
   if ids.any():coef[ids,j]=np.linalg.solve(Z[:,ids].T@Z[:,ids]+ridge*np.eye(ids.sum()),Z[:,ids].T@B[:,j])
  if threshold:
   active &= abs(coef)>=threshold;coef[~active]=0.
 singular=np.linalg.svd(Z,compute_uv=False)
 return dict(library=library,ridge=ridge,threshold=threshold,mean=m.tolist(),std=s.tolist(),coefficient=(coef/scale[:,None]).tolist(),integral_column_rms=scale.tolist(),singular_values=singular.tolist(),rows=len(A),active_coefficients=int(np.count_nonzero(coef)),normalized_integral_fit_mse=float(np.mean((Z@coef-B)**2)))
def field(model):
 m=np.asarray(model['mean']);s=np.asarray(model['std']);coef=np.asarray(model['coefficient'])
 return lambda x:features(np.asarray(x),m,s,model['library'])@coef*s
def integrate(fn,y0,t):
 count=0;started=time.perf_counter();status=None;prediction=None
 def f(time,x):
  nonlocal count
  count+=1
  if count>20000:raise RuntimeError('Declared20000-RHS-call budget exceeded')
  if not np.isfinite(x).all() or abs(x).max()>1e6:raise RuntimeError('Declared1e6 state bound exceeded, no clipping')
  out=fn(x)
  if not np.isfinite(out).all():raise RuntimeError('Nonfinite predicted vector field')
  return out
 try:
  sol=solve_ivp(f,(float(t[0]),float(t[-1])),y0,t_eval=t,method='RK45',rtol=1e-4,atol=1e-6);prediction=sol.y.T;status=dict(success=bool(sol.success and len(sol.t)==len(t)),message=str(sol.message),status=int(sol.status),nfev=int(sol.nfev),returned_steps=len(sol.t))
 except Exception as exc:status=dict(success=False,message=repr(exc),status=None,nfev=count,returned_steps=0)
 status['wall_seconds']=time.perf_counter()-started;return prediction,status
def metric(p,y):
 valid=p is not None and p.shape==y.shape and np.isfinite(p).all()
 if not valid:return dict(finite_complete=False,mse=None,rmse=None,mae=None,nrmse=None)
 e=p-y;mse=float(np.mean(e*e));return dict(finite_complete=True,mse=mse,rmse=float(np.sqrt(mse)),mae=float(np.mean(abs(e))),nrmse=float(np.sqrt(mse)/np.std(y)))
def truth_field(meta):
 vars=sp.symbols('x_0 x_1');constants={f'c_{i}':float(c) for i,c in enumerate(meta['consts'])};expressions=[sp.sympify(e.replace('^','**'),locals={**constants,**dict(zip(['x_0','x_1'],vars))}) for e in meta['eq'].split('|')];f=sp.lambdify(vars,expressions,'numpy');return lambda x:np.asarray(f(*x),dtype=np.float64)
def main():
 started=time.perf_counter();native=read(OUT/'NATIVE_RESULTS80_V2.json');art=storage('artifact')/'ordinary_context_v2';art.mkdir(exist_ok=False);rows=[];candidates=[];models=[]
 dump(OUT/'CONTROL_OPENING80_V2.json',dict(utc=utc(),protocol_sha256=sha(ROOT/'ORDINARY_CONTROL80_V1.md'),native_result_sha256=sha(OUT/'NATIVE_RESULTS80_V2.json'),native_scores_already_inspected=True,validation='Remaining65 noisy context points, not clean outer trajectories',source_files_checked=check_source(),new_neural_fits=0))
 for sid in [26,42,28]:
  nr=next(x for x in native['rows'] if x['system']==sid and x['initial_condition_index']==0);assert sha(nr['array_path'])==nr['array_sha256']
  with np.load(nr['array_path'],allow_pickle=False) as a:x=a['context'][0].copy();t=a['context_times'][0,:,0].copy()
  assert x.shape==(256,2);this=[]
  for library,ridge,threshold in itertools.product(['polynomial3','polynomial3_trig'],[1e-6,1e-3,1.],[0.,.03]):
   model=fit(t[:192],x[:192],library,ridge,threshold);p,solver=integrate(field(model),x[191],t[191:]);scores=metric(p,x[191:]);row=dict(system=sid,library=library,ridge=ridge,threshold=threshold,validation=scores,solver=solver,model=model);this.append(row);candidates.append(row)
  valid=[z for z in this if z['solver']['success'] and z['validation']['finite_complete']];assert valid,('All context candidates failed, preserve and investigate',sid)
  chosen=min(valid,key=lambda z:(z['validation']['mse'],z['library'],z['ridge'],z['threshold']));model=fit(t,x,chosen['library'],chosen['ridge'],chosen['threshold']);modelpath=art/f'system{sid}_weak_integral_model_v1.json';dump(modelpath,dict(system=sid,selected_inner_context_config={key:chosen[key] for key in ['library','ridge','threshold']},model=model));models.append(dict(system=sid,path=str(modelpath),sha256=sha(modelpath)))
  meta=next(z['meta'] for z in native['qualification_checks'] if z['system']==sid)
  for k in [0,1]:
   base=next(z for z in native['rows'] if z['system']==sid and z['initial_condition_index']==k);assert sha(base['array_path'])==base['array_sha256']
   with np.load(base['array_path'],allow_pickle=False) as a:y=a['target'].copy();timegrid=a['time'].copy()
   for name,fn in [('ordinary_weak_integral',field(model)),('privileged_true_equations',truth_field(meta))]:
    p,solver=integrate(fn,y[0].copy(),timegrid);path=art/f'system{sid}_ic{k}_{name}.npz';np.savez_compressed(path,prediction=p if p is not None else np.empty((0,2)),target=y,time=timegrid,context=x,context_times=t)
    row=dict(system=sid,initial_condition_index=k,role=base['role'],method=name,privileged=name=='privileged_true_equations',**metric(p,y),solver=solver,native_fim_mse=base['mse'],native_fim_nrmse=base['nrmse'],selected_library=chosen['library'] if name=='ordinary_weak_integral' else None,array_path=str(path),array_sha256=sha(path));rows.append(row);print(json.dumps({key:row[key] for key in ['system','initial_condition_index','method','mse','nrmse','solver']}),flush=True)
  state('CONTROL_PROGRESS80_V2.json',dict(status='ORDINARY_CONTEXT_CONTROLS_RUNNING',system=sid,closed_rows=len(rows),elapsed_seconds=time.perf_counter()-started))
 dump(OUT/'ORDINARY_RESULTS80_V2.json',dict(utc=utc(),status='ORDINARY_AND_TRUE_EQUATION_CONTROLS_CLOSED_UNVERIFIED',rows=rows,candidates=candidates,models=models,inner_candidate_fits=len(candidates),full_context_refits=len(models),new_neural_fits=0,post_discovery_not_independent_confirmation=True,source_informed_trig_library=True,accepted_final_directions=0,overall_goal='ACTIVE_UNACHIEVED',wall_seconds=time.perf_counter()-started));assert len(rows)==12 and len(candidates)==36 and len(models)==3
if __name__=='__main__':main()
