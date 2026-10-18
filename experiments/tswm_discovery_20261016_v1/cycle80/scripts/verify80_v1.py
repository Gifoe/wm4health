"""Independent context/metric reconstruction and cached-native integration replays."""
import sys,pathlib,time,gc
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from native80_v1 import *
sys.path.insert(0,str(PROJECT/'cycle71/scripts'))
from runtime_affinity71_v1 import configure
AFFINITY=configure('efficient')
import numpy as np
import torch
torch.set_num_threads(2);torch.set_num_interop_threads(2)
from fim.models.ode import load_fim_ode_local
from ode_tutorial_helper import make_fim_vf_fn
from scipy.integrate import solve_ivp
def independent_features(x,model):
 x=np.asarray(x);z=(x-np.asarray(model['mean']))/np.asarray(model['std']);p=[]
 for degree in range(4):
  for i in range(degree+1):p.append(z[...,0]**i*z[...,1]**(degree-i))
 if model['library']=='polynomial3_trig':p.extend([np.sin(x[...,0]),np.sin(x[...,1]),np.cos(x[...,0]),np.cos(x[...,1])])
 return np.stack(p,axis=-1)
def independent_field(x,model):return independent_features(x,model)@np.asarray(model['coefficient'])*np.asarray(model['std'])
def independent_weak_design(t,x,model):
 theta=independent_features(x,model);z=(x-np.asarray(model['mean']))/np.asarray(model['std']);aa=[];bb=[]
 for width in [8,16,32]:
  for j in range(len(x)-width):aa.append(np.trapz(theta[j:j+width+1],t[j:j+width+1],axis=0));bb.append(z[j+width]-z[j])
 return np.asarray(aa),np.asarray(bb)
def main():
 started=time.perf_counter();native=read(OUT/'NATIVE_RESULTS80_V2.json');control=read(OUT/'ORDINARY_RESULTS80_V1.json');asset=assets();database=read(next(x['path'] for x in asset if x['relative'].endswith('.json') and not x['relative'].endswith('config.json')));metricmax=0.;values=0;candidate_max=0.;checks=[];regression=[]
 for row in native['rows']+control['rows']:
  assert sha(row['array_path'])==row['array_sha256']
  with np.load(row['array_path'],allow_pickle=False) as a:p=a['prediction'].copy();y=a['target'].copy()
  assert p.shape==y.shape==(512,2) and np.isfinite(p).all();values+=p.size+y.size;difference=p-y;mse=float(np.sum(difference*difference)/difference.size);rmse=float(np.sqrt(mse));nrmse=rmse/float(np.std(y))
  for key,val in [('mse',mse),('rmse',rmse),('mae',float(np.mean(abs(difference)))),('nrmse',nrmse)]:metricmax=max(metricmax,abs(row[key]-val));assert abs(row[key]-val)<1e-12
  entry=next(x for x in database if x['id']==row['system']);solutions=[s for s in entry['solutions'][0] if s.get('success',True)];assert np.array_equal(y,np.asarray(solutions[row['initial_condition_index']]['y']).T)
 for sid in [26,42,28]:
  row=next(x for x in native['rows'] if x['system']==sid and x['initial_condition_index']==0)
  with np.load(row['array_path'],allow_pickle=False) as a:context=a['context'][0].copy();ct=a['context_times'][0,:,0].copy()
  entry=next(x for x in database if x['id']==sid);raw=next(s for s in entry['solutions'][0] if s.get('success',True));original=np.asarray(raw['y']).T;idx=np.linspace(0,len(original)-1,256,dtype=int);expected=original[idx]*(1+.03*np.random.default_rng(0).standard_normal((256,2)));assert np.array_equal(context,expected);assert np.array_equal(ct,np.asarray(raw['t'])[idx])
  for candidate in [c for c in control['candidates'] if c['system']==sid]:
   model=candidate['model'];A,B=independent_weak_design(ct[:192],context[:192],model);scale=np.maximum(np.sqrt(np.mean(A*A,axis=0)),1e-10);Z=A/scale;C=np.asarray(model['coefficient'])*scale[:,None];loss=float(np.mean((Z@C-B)**2));assert abs(loss-model['normalized_integral_fit_mse'])<1e-10
   if candidate['solver']['success']:
    sol=solve_ivp(lambda t,x:independent_field(x,model),(ct[191],ct[-1]),context[191],t_eval=ct[191:],method='RK45',rtol=1e-4,atol=1e-6);assert sol.success;score=float(np.mean((sol.y.T-context[191:])**2));error=abs(score-candidate['validation']['mse']);candidate_max=max(candidate_max,error);assert error<1e-7
  record=next(x for x in control['models'] if x['system']==sid);assert sha(record['path'])==record['sha256'];model=read(record['path'])['model'];A,B=independent_weak_design(ct,context,model);scale=np.maximum(np.sqrt(np.mean(A*A,axis=0)),1e-10);Z=A/scale;active=np.ones((Z.shape[1],2),dtype=bool);C=np.zeros_like(active,dtype=np.float64)
  for _ in range(5):
   for j in range(2):
    selected=np.flatnonzero(active[:,j]);aug=np.vstack([Z[:,selected],np.sqrt(model['ridge'])*np.eye(len(selected))]);rhs=np.concatenate([B[:,j],np.zeros(len(selected))]);C[selected,j]=np.linalg.lstsq(aug,rhs,rcond=None)[0]
   if model['threshold']:active &= abs(C)>=model['threshold'];C[~active]=0.
  actual=np.asarray(model['coefficient']);maxdiff=float(abs(C/scale[:,None]-actual).max());assert np.allclose(C/scale[:,None],actual,rtol=1e-5,atol=1e-7);regression.append(dict(system=sid,augmented_lstsq_coefficient_max_abs_difference=maxdiff,normalized_design_smallest_singular_value=float(np.linalg.svd(Z,compute_uv=False)[-1]),selected_library=model['library']))
 weights=next(x['path'] for x in asset if x['relative'].endswith('.safetensors'));model=load_fim_ode_local(pathlib.Path(weights).parent,device='cpu');model.eval()
 for sid in [26,42,28]:
  row=next(x for x in native['rows'] if x['system']==sid and x['initial_condition_index']==0)
  with np.load(row['array_path'],allow_pickle=False) as a:context=a['context'].copy();ct=a['context_times'].copy();query=a['query'].copy();savedfield=a['cached_field'].copy()
  fn=make_fim_vf_fn(model,context,ct,'cpu');difference=float(abs(fn(query)-savedfield).max());assert difference<2e-5
  for k in [0,1]:
   row=next(x for x in native['rows'] if x['system']==sid and x['initial_condition_index']==k)
   with np.load(row['array_path'],allow_pickle=False) as a:t=a['time'].copy();saved=a['prediction'].copy();y0=a['initial_state'].copy()
   sol=solve_ivp(lambda t,x:fn(x[None])[0],(float(t[0]),float(t[-1])),y0,t_eval=t,method='RK45',rtol=1e-4,atol=1e-6);assert sol.success and sol.y.T.shape==saved.shape;error=float(abs(sol.y.T-saved).max());assert np.allclose(sol.y.T,saved,atol=2e-5,rtol=2e-5);checks.append(dict(system=sid,initial_condition=k,max_abs_cached_vs_native_integrated_difference=error,repeated_query_max_abs=difference,nfev=sol.nfev))
 assert len(checks)==6 and values==36864 and len(regression)==3 and check_source()==70
 dump(OUT/'VERIFICATION80_V1.json',dict(utc=utc(),status='PASS_ALL18_METRICS_CLEAN_CONTEXT_RECONSTRUCTION36_SELECTIONS3_COEFFICIENT_REPLAYS6_NATIVE_INTEGRATIONS',native_result_sha256=sha(OUT/'NATIVE_RESULTS80_V2.json'),control_result_sha256=sha(OUT/'ORDINARY_RESULTS80_V1.json'),rows=18,prediction_and_target_values=values,metric_max_abs_error=metricmax,context_candidate_validation_max_abs_error=candidate_max,regression_checks=regression,native_replays=checks,wall_seconds=time.perf_counter()-started,source_files=70,accepted_final_directions=0))
 print(json.dumps(dict(status='PASS',rows=18,prediction_and_target_values=values,metric_max_abs_error=metricmax,candidate_validation_max_abs_error=candidate_max,native_integration_max_abs_error=max(x['max_abs_cached_vs_native_integrated_difference'] for x in checks))),flush=True)
if __name__=='__main__':main()
