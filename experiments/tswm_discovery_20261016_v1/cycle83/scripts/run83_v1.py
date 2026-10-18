import sys,pathlib,time,itertools
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common83_v1 import *
import data83_v1 as data
sys.path.insert(0,str(PROJECT/'cycle80/scripts'))
import native80_v1 as old
import ordinary80_v1 as ordinary
import numpy as np
import torch
from scipy.integrate import solve_ivp
from scipy.spatial import cKDTree
from fim.models.ode import load_fim_ode_local
from ode_tutorial_helper import make_fim_vf_fn
torch.set_num_threads(2);torch.set_num_interop_threads(2)
def powers(d):return sorted([p for p in itertools.product(range(4),repeat=d) if sum(p)<=3],key=lambda p:(sum(p),p))
def phi(x,m,s,kind):
 z=(np.asarray(x)-m)/s;d=z.shape[-1]
 if kind=='weak_polynomial3':return np.stack([np.prod(z**np.asarray(p),axis=-1) for p in powers(d)],axis=-1)
 rng=np.random.default_rng(202683);w=rng.normal(size=(d,128));phase=rng.uniform(0,2*np.pi,128);return np.concatenate([np.sqrt(2/128)*np.cos(z@w+phase),np.ones((*z.shape[:-1],1))],-1)
def fit(x,t,kind,ridge):
 m=x.mean((0,1));s=np.maximum(x.std((0,1)),1e-8);h=phi(x,m,s,kind);ip=np.concatenate([np.zeros((len(x),1,h.shape[-1])),np.cumsum(.5*(h[:,1:]+h[:,:-1])*np.diff(t)[None,:,None],axis=1)],1);z=(x-m)/s;aa=[];bb=[]
 for width in [8,16,32]:aa.append((ip[:,width:]-ip[:,:-width]).reshape(-1,h.shape[-1]));bb.append((z[:,width:,0]-z[:,:-width,0]).reshape(-1))
 a=np.concatenate(aa);b=np.concatenate(bb);scale=np.maximum(np.sqrt(np.mean(a*a,0)),1e-10);az=a/scale;c=np.linalg.solve(az.T@az+ridge*np.eye(a.shape[-1]),az.T@b)/scale
 return dict(kind=kind,ridge=ridge,mean=m.tolist(),std=s.tolist(),coefficient=c.tolist(),weak_column_rms=scale.tolist(),weak_rows=len(a),context_shape=list(x.shape),parameters=len(c))
def ordinary_field(pars,parents):
 values=[(np.asarray(p['mean']),np.asarray(p['std']),np.asarray(p['coefficient']),p['kind']) for p in pars]
 def fn(q):return np.stack([phi(np.asarray(q)[...,idx],m,s,kind)@c*s[0] for idx,(m,s,c,kind) in zip(parents,values)],axis=-1)
 return fn
class Native:
 def __init__(self,model,x,t,parents):
  self.model=model;self.parents=parents;self.dim=len(parents);ctx=np.stack([np.pad(x[...,p],[(0,0),(0,0),(0,3-len(p))]) for p in parents]);times=np.broadcast_to(t[None,None,:,None],(*ctx.shape[:-1],1)).copy();tensor=torch.tensor(ctx,dtype=torch.float32);tt=torch.tensor(times,dtype=torch.float32);mask=torch.ones((*tensor.shape[:-1],1),dtype=torch.bool)
  with torch.no_grad():self.wrapped,self.mask,self.concept=model.trajectory_encoding(tensor,tt,mask)
 def __call__(self,q):
  q=np.asarray(q);loc=np.stack([np.pad(q[...,p],[(0,0),(0,3-len(p))]) for p in self.parents]);tensor=torch.tensor(loc,dtype=torch.float32)
  with torch.no_grad():norm=self.model.spatial_norm.normalization_map(tensor,self.concept._states_norm_stats);out=self.model.function_decoding(norm,self.mask,self.wrapped,self.concept);prediction=self.model.get_prediction_for_eval(out)[:,:,0].T
  return prediction.numpy()
def simulate(fn,y0,t,clamp=False):
 started=time.perf_counter();calls=0
 def rhs(now,y):
  nonlocal calls
  calls+=1
  if calls>20000:raise RuntimeError('Declared 20000 RHS budget exceeded')
  if not np.isfinite(y).all() or abs(y).max()>1e6:raise RuntimeError('Declared absolute state bound1e6 exceeded; no clipping')
  q=y.copy()
  if clamp:q[0]=data.plan(now,y0[0])
  value=fn(q[None])[0]
  if clamp:value[0]=.4*np.cos(.8*now)
  if not np.isfinite(value).all():raise RuntimeError('Nonfinite vector field')
  return value
 try:
  sol=solve_ivp(rhs,(float(t[0]),float(t[-1])),y0,t_eval=t,method='RK45',rtol=1e-4,atol=1e-6);p=sol.y.T
  if clamp:p[:,0]=data.plan(sol.t,y0[0])
  st=dict(success=bool(sol.success and len(sol.t)==len(t)),nfev=sol.nfev,returned_steps=len(sol.t),message=str(sol.message))
 except Exception as exc:p=np.empty((0,len(y0)));st=dict(success=False,nfev=calls,returned_steps=0,message=repr(exc))
 return p,dict(**st,wall_seconds=time.perf_counter()-started)
def metrics(p,y,clamp=False):
 if p is None or p.shape!=y.shape or not np.isfinite(p).all():return dict(finite_complete=False,mse=None,nrmse=None,coordinate_mean_nrmse=None,coordinate_nrmse=[])
 pp=p[:,1:] if clamp else p;yy=y[:,1:] if clamp else y;e=pp-yy;coord=np.sqrt(np.mean(e*e,0))/np.maximum(yy.std(0),1e-12)
 return dict(finite_complete=True,mse=float(np.mean(e*e)),nrmse=float(np.sqrt(np.mean(e*e))/max(yy.std(),1e-12)),coordinate_mean_nrmse=float(coord.mean()),coordinate_nrmse=coord.tolist())
def coverage(x,p,parents):
 if p.ndim!=2 or len(p)==0 or not np.isfinite(p).all():return None
 dd=[]
 for idx in parents:
  context=x[...,idx].reshape(-1,len(idx));m=context.mean(0);s=np.maximum(context.std(0),1e-8);dist=cKDTree((context-m)/s).query((p[:,idx]-m)/s)[0];dd.extend(dist.tolist())
 return dict(mean=float(np.mean(dd)),p95=float(np.quantile(dd,.95)),maximum=float(max(dd)))
def qualify(model,x,t,parents):
 q=x[0,np.linspace(0,255,17,dtype=int)];batch=Native(model,x,t,parents)(q);rows=[];tt=np.broadcast_to(t[None,:,None],(4,len(t),1)).copy()
 for i,idx in enumerate(parents):
  ref=make_fim_vf_fn(model,x[...,idx],tt,'cpu')(q[:,idx])[:,0];err=float(abs(batch[:,i]-ref).max());assert np.allclose(batch[:,i],ref,atol=2e-5,rtol=2e-5),(i,err);rows.append(dict(target=i,dimension=len(idx),max_abs=err))
 return rows
def main():
 started=time.perf_counter();manifest=read(OUT/'DATA_MANIFEST83_V1.json');assert manifest['protocol_sha256']==sha(ROOT/'PROTOCOL83_V1.md');assert old.check_source()==70;assets=old.assets();weight=next(z for z in assets if z['relative'].endswith('.safetensors'));model=load_fim_ode_local(pathlib.Path(weight['path']).parent,'cpu');model.eval()
 for p in model.parameters():p.requires_grad_(False)
 qualifications=[]
 for family in ['fhn','hill']:
  it=next(z for z in manifest['rows'] if z['family']==family)
  with np.load(it['path'],allow_pickle=False) as a:qualifications.append(dict(family=family,rows=qualify(model,a['context'],a['time'],it['parents'])))
 art=storage('artifact')/'composition_screen_v1';art.mkdir(exist_ok=False);dump(OUT/'OPENING83_V1.json',dict(utc=utc(),protocol_sha256=sha(ROOT/'PROTOCOL83_V1.md'),data_manifest_sha256=sha(OUT/'DATA_MANIFEST83_V1.json'),qualification=qualifications,weights_sha256=weight['sha256'],source_files_checked=70,affinity=ordinary.AFFINITY,device='cpu',new_neural_training_fits=0));fields=[];rollouts=[];responses=[];candidates=[];models=[];truth_checks=[]
 for item in manifest['rows']:
  tag=item['tag'];par=item['parents'];assert sha(item['path'])==item['sha256']
  with np.load(item['path'],allow_pickle=False) as a:x=a['context'].copy();t=a['time'].copy();q=a['queries'].copy();fy=a['field_target'].copy();test_initial=a['test_initial_states'].copy()
  state('PROGRESS83_V1.json',dict(status='RUNNING',tag=tag,rollouts=len(rollouts),elapsed_seconds=time.perf_counter()-started));methods={'FIM_local_composed':Native(model,x,t,par)}
  for kind in ['weak_polynomial3','weak_rff128']:
   begin=time.perf_counter();these=[]
   for ridge in [.001,.1,10.]:
    pars=[fit(x[:,:192,idx],t[:192],kind,ridge) for idx in par];fn=ordinary_field(pars,par);loss=[];st=[]
    for j in range(4):
     pred,status=simulate(fn,x[j,191],t[191:]);st.append(status)
     if status['success']:loss.append(float(np.mean((pred-x[j,191:])**2)))
    row=dict(tag=tag,kind=kind,ridge=ridge,validation_mse=float(np.mean(loss)) if len(loss)==4 else None,successful_validation_paths=len(loss),solvers=st);these.append(row);candidates.append(row)
   valid=[r for r in these if r['validation_mse'] is not None]
   if not valid:methods[kind]=None;models.append(dict(tag=tag,kind=kind,all_candidates_failed=True));continue
   chosen=min(valid,key=lambda r:(r['validation_mse'],r['ridge']));pars=[fit(x[...,idx],t,kind,chosen['ridge']) for idx in par];mp=art/f'{tag}_{kind}_model.json';dump(mp,dict(parents=par,locals=pars,ridge=chosen['ridge'],adaptation_seconds=time.perf_counter()-begin));models.append(dict(tag=tag,kind=kind,path=str(mp),sha256=sha(mp)));methods[kind]=ordinary_field(pars,par)
  grid=np.linspace(0,8,257);refs={}
  for j,y0 in enumerate(test_initial):
   for clamp in [False,True]:
    ref=data.reference(item['family'],item['setting'],y0,grid,clamp);check,status=simulate(lambda z:data.true_field(z,item['family'],item['setting']),y0,grid,clamp);assert status['success'];score=metrics(check,ref,clamp);assert score['coordinate_mean_nrmse']<.005;truth_checks.append(dict(tag=tag,initial_index=j,clamp=clamp,score=score,solver=status));refs[j,clamp]=ref
  for kind,fn in methods.items():
   fp=fn(q) if fn is not None else np.empty((0,len(par)));path=art/f'{tag}_{kind}_field.npz';np.savez_compressed(path,queries=q,target=fy,prediction=fp);fields.append(dict(tag=tag,family=item['family'],setting=item['setting'],seed=item['seed'],noise_sigma=item['noise_sigma'],kind=kind,**metrics(fp,fy),path=str(path),sha256=sha(path)));these={}
   for j,y0 in enumerate(test_initial):
    for clamp in [False,True]:
     ref=refs[j,clamp]
     if fn is None:pred=np.empty((0,len(par)));st=dict(success=False,nfev=0,returned_steps=0,message='All context candidates failed',wall_seconds=0.)
     else:pred,st=simulate(fn,y0,grid,clamp)
     rp=art/f'{tag}_{kind}_ic{j}_clamp{int(clamp)}.npz';np.savez_compressed(rp,time=grid,initial_state=y0,target=ref,prediction=pred);row=dict(tag=tag,family=item['family'],setting=item['setting'],seed=item['seed'],noise_sigma=item['noise_sigma'],kind=kind,initial_index=j,clamp=clamp,**metrics(pred,ref,clamp),solver=st,reference_support=coverage(x,ref,par),prediction_support=coverage(x,pred,par),path=str(rp),sha256=sha(rp));rollouts.append(row);these[j,clamp]=pred
    free=these[j,False];clamped=these[j,True];valid=free.shape==refs[j,False].shape and clamped.shape==refs[j,True].shape;response=metrics(clamped-free if valid else None,refs[j,True]-refs[j,False],True);responses.append(dict(tag=tag,family=item['family'],noise_sigma=item['noise_sigma'],kind=kind,initial_index=j,**response))
   print(json.dumps(dict(type='method_closed',tag=tag,kind=kind,trajectory_records=len(rollouts),elapsed_seconds=time.perf_counter()-started)),flush=True)
 assert len(fields)==48 and len(rollouts)==192 and len(responses)==96 and len(candidates)==96 and old.check_source()==70 and sha(weight['path'])==weight['sha256'];dump(OUT/'RESULTS83_V1.json',dict(utc=utc(),status='COMPOSITION_SCREEN_CLOSED_UNVERIFIED',fields=fields,rollouts=rollouts,responses=responses,candidates=candidates,models=models,true_solver_checks=truth_checks,context_tasks=16,new_neural_training_fits=0,evaluation_is_frozen_development=True,weights_unchanged=True,wall_seconds=time.perf_counter()-started,accepted_final_directions=0,overall_goal='ACTIVE_UNACHIEVED'));state('PROGRESS83_V1.json',dict(status='COMPOSITION_SCREEN_CLOSED_UNVERIFIED',fields=48,rollouts=192,wall_seconds=time.perf_counter()-started))
if __name__=='__main__':main()
