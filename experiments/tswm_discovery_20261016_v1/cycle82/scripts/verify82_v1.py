"""Recompute source identities, scores, selected weak fits, and all native field queries."""
import sys,pathlib,math,time
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common82_v1 import *
import run82_v1 as run
import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial import Delaunay,cKDTree

def main():
 started=time.perf_counter();r=read(OUT/'RESULTS82_V1.json');manifest=read(OUT/'DATA_MANIFEST82_V1.json');assert len(r['fields'])==64 and len(r['rollouts'])==128;data={};identity_count=0;score_error=0.;scalars=0;failures=[];selections=[];coefs=[];native=[];ordinary=[];true_replays=[]
 assets=run.old.assets();weights=next(z for z in assets if z['relative'].endswith('.safetensors'));model=run.load_fim_ode_local(pathlib.Path(weights['path']).parent,'cpu');model.eval();models={(z['tag'],z['kind']):z for z in r['models']}
 for item in manifest['rows']:
  tag=f"m{item['mechanism']}_seed{item['seed']}_{item['regime']}";assert sha(item['path'])==item['sha256']
  with np.load(item['path'],allow_pickle=False) as a:data[tag]={key:a[key].copy() for key in a.files}
  a=data[tag];assert np.array_equal(a['field_target'],run.target(a['queries'],item['parameters']));assert a['query_distance'].max()<=.1;identity_count+=2
  for j,initial in enumerate(a['initial_states']):
   s=solve_ivp(lambda t,y:run.target(y,item['parameters'])*0 if False else np.array([run.target(y,item['parameters']),-.4*y[1]-.1*y[1]**3+item['k']*y[2],-.3*y[2]+.05*y[0]]),(0,8),initial,t_eval=a['time'],method='DOP853',rtol=1e-9,atol=1e-11);assert s.success and np.array_equal(s.y.T,a['context'][j]);identity_count+=1
  tt=np.broadcast_to(a['time'][None,:,None],(8,256,1)).copy()
  for kind,dim in [('FIM_local2',2),('FIM_full3_diagnostic',3)]:
   row=next(z for z in r['fields'] if z['tag']==tag and z['kind']==kind);q=a['queries'];query=q if dim==2 else np.c_[q,np.zeros(len(q))];vf=run.make_fim_vf_fn(model,a['context'][:,:,:dim],tt,'cpu');prediction=vf(query)[:,0]
   with np.load(row['path'],allow_pickle=False) as stored:difference=float(abs(prediction-stored['prediction']).max());assert difference==0.;native.append(dict(tag=tag,kind=kind,maximum_field_replay_error=difference))
  for kind in ['weak_polynomial3','weak_rff128']:
   candidates=[z for z in r['candidates'] if z['tag']==tag and z['kind']==kind];valid=[z for z in candidates if z['validation_mse'] is not None];saved=models[tag,kind]
   if not valid:assert saved['all_candidates_failed'];continue
   assert sha(saved['path'])==saved['sha256'];pars=read(saved['path']);best=min(valid,key=lambda z:(z['validation_mse'],z['ridge']));assert best['ridge']==pars['ridge'];selections.append(dict(tag=tag,kind=kind,selected_ridge=pars['ridge'],candidate_count=len(candidates)));x=a['context'][:,:,:2];t=a['time'];m=x.mean((0,1));std=x.std((0,1));H=run.phi(x,m,std,kind);z=(x-m)/std;aa=[];bb=[]
   for width in [8,16,32]:
    for j in range(8):
     for start in range(len(t)-width):
      end=start+width;aa.append(np.trapz(H[j,start:end+1],t[start:end+1],axis=0));bb.append(z[j,end,0]-z[j,start,0])
   A=np.asarray(aa);B=np.asarray(bb);scale=np.maximum(np.sqrt(np.mean(A*A,axis=0)),1e-10);Z=A/scale;ridge=pars['ridge'];ind=np.linalg.lstsq(np.r_[Z,np.sqrt(ridge)*np.eye(Z.shape[1])],np.r_[B,np.zeros(Z.shape[1])],rcond=None)[0]/scale;stored=np.asarray(pars['coefficient']);diff=float(abs(ind-stored).max());assert np.allclose(ind,stored,atol=1e-6,rtol=1e-5),(tag,kind,diff);coefs.append(dict(tag=tag,kind=kind,coefficient_max_abs=diff,coefficient_count=len(ind)))
   fr=next(z for z in r['fields'] if z['tag']==tag and z['kind']==kind)
   with np.load(fr['path'],allow_pickle=False) as f:dp=float(abs(run.field(pars)(f['queries'])-f['prediction']).max());assert dp==0.;ordinary.append(dict(tag=tag,kind=kind,stored_field_replay_max_abs=dp))
 for row in r['fields']+r['rollouts']:
  assert sha(row['path'])==row['sha256']
  with np.load(row['path'],allow_pickle=False) as a:
   p=a['prediction'];y=a['target'];valid=p.shape==y.shape and np.isfinite(p).all();assert valid==row['finite_complete'];scalars+=p.size+y.size
   if valid:
    e=p-y;values=dict(mse=float(np.mean(e*e)),rmse=float(np.sqrt(np.mean(e*e))),mae=float(abs(e).mean()),nrmse=float(np.sqrt(np.mean(e*e))/y.std()))
    for key,val in values.items():score_error=max(score_error,abs(row[key]-val));assert math.isclose(row[key],val,abs_tol=1e-12,rel_tol=1e-12)
   else:failures.append(dict(tag=row['tag'],kind=row['kind'],plan=row.get('plan_index'),solver=row.get('solver')))
   if 'plan_index' in row:
    t=a['time'];om=row['omega'];u=lambda now:.75*np.sin(om*now)+.25*np.cos(.5*om*now);assert np.array_equal(a['plan'],u(t)) and a['initial_state'][0]==row['initial_state'];item=next(i for i in manifest['rows'] if i['mechanism']==row['mechanism'] and i['seed']==row['seed'] and i['regime']==row['regime']);sol=solve_ivp(lambda now,state:[run.target(np.array([state[0],u(now)]),item['parameters'])],(0,6),[row['initial_state']],t_eval=t,method='DOP853',rtol=1e-9,atol=1e-11);assert sol.success and np.array_equal(sol.y.T,a['target']);identity_count+=3
 for mechanism in range(4):
  for seed in [2026821,2026822]:
   closed=data[f'm{mechanism}_seed{seed}_closed'];opened=data[f'm{mechanism}_seed{seed}_open'];assert np.array_equal(closed['queries'],opened['queries']) and np.array_equal(closed['field_target'],opened['field_target']) and np.array_equal(closed['initial_states'],opened['initial_states']);identity_count+=3
 assert len(native)==32 and run.old.check_source()==70 and sha(weights['path'])==weights['sha256'];dump(OUT/'VERIFICATION82_V1.json',dict(utc=utc(),status='PASS_IDENTITIES_SCORES_NATIVE_FIELDS_AND_INDEPENDENT_WEAK_FITS',score_max_abs=score_error,prediction_target_scalars=scalars,identity_arrays_checked=identity_count,native_field_replays=native,ordinary_field_replays=ordinary,independent_coefficient_checks=coefs,selection_checks=selections,failed_records=failures,weights_unchanged=True,source_files_checked=70,wall_seconds=time.perf_counter()-started));print(json.dumps(dict(status='PASS',score_max_abs=score_error,native_field_replays=len(native),independent_coefficients=sum(z['coefficient_count'] for z in coefs),maximum_coefficient_difference=max(z['coefficient_max_abs'] for z in coefs),failed_records=len(failures))),flush=True)
if __name__=='__main__':main()
