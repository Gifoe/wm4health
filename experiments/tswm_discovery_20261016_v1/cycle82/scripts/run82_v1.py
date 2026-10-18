import sys,pathlib,time,itertools
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common82_v1 import *
from data82_v1 import target
sys.path.insert(0,str(PROJECT/'cycle80/scripts'))
import native80_v1 as old
import ordinary80_v1 as ordinary
import numpy as np
import torch
from scipy.integrate import solve_ivp
from fim.models.ode import load_fim_ode_local
from ode_tutorial_helper import make_fim_vf_fn
torch.set_num_threads(2);torch.set_num_interop_threads(2)
POWERS=[(i,deg-i) for deg in range(4) for i in range(deg+1)]
def phi(x,m,s,kind):
 z=(np.asarray(x)-m)/s
 if kind=='weak_polynomial3':return np.stack([z[...,0]**i*z[...,1]**j for i,j in POWERS],-1)
 rng=np.random.default_rng(202682);W=rng.normal(size=(2,128));phase=rng.uniform(0,2*np.pi,size=128);return np.concatenate([np.sqrt(2/128)*np.cos(z@W+phase),np.ones((*z.shape[:-1],1))],axis=-1)
def fit(x,t,kind,ridge):
 m=x.mean((0,1));s=np.maximum(x.std((0,1)),1e-8);H=phi(x,m,s,kind);ip=np.concatenate([np.zeros((len(x),1,H.shape[-1])),np.cumsum(.5*(H[:,1:]+H[:,:-1])*np.diff(t)[None,:,None],axis=1)],axis=1);z=(x-m)/s;A=[];B=[]
 for w in [8,16,32]:A.append((ip[:,w:]-ip[:,:-w]).reshape(-1,H.shape[-1]));B.append((z[:,w:,0]-z[:,:-w,0]).reshape(-1,1))
 A=np.concatenate(A);B=np.concatenate(B);scale=np.maximum(np.sqrt(np.mean(A*A,0)),1e-10);Z=A/scale;coef=np.linalg.solve(Z.T@Z+ridge*np.eye(Z.shape[1]),Z.T@B)[:,0]/scale;sv=np.linalg.svd(Z,compute_uv=False);pars=dict(kind=kind,ridge=ridge,mean=m.tolist(),std=s.tolist(),coefficient=coef.tolist(),weak_column_rms=scale.tolist(),normalized_design_singular_values=sv.tolist(),weak_rows=len(A),context_paths=len(x),context_steps=x.shape[1],parameters=len(coef));return pars
def field(pars):
 m=np.asarray(pars['mean']);s=np.asarray(pars['std']);c=np.asarray(pars['coefficient']);return lambda q:phi(q,m,s,pars['kind'])@c*s[0]
def controlled(fn,u,y0,t):
 return ordinary.integrate(lambda y:np.zeros(1),np.array([y0]),t) if False else solve_control(fn,u,y0,t)
def solve_control(fn,u,y0,t):
 count=0;started=time.perf_counter()
 def rhs(time,y):
  nonlocal count
  count+=1
  if count>20000:raise RuntimeError('Declared20000-RHS-call budget exceeded')
  if not np.isfinite(y).all() or abs(y).max()>1e6:raise RuntimeError('Declared1e6 absolute-state bound exceeded; no clipping')
  value=float(fn(np.array([[y[0],u(time)]]))[0])
  if not np.isfinite(value):raise RuntimeError('Nonfinite field')
  return np.array([value])
 try:
  sol=solve_ivp(rhs,(float(t[0]),float(t[-1])),[y0],t_eval=t,method='RK45',rtol=1e-4,atol=1e-6);p=sol.y.T;status=dict(success=bool(sol.success and len(sol.t)==len(t)),nfev=sol.nfev,message=str(sol.message),returned_steps=len(sol.t))
 except Exception as exc:p=np.empty((0,1));status=dict(success=False,nfev=count,message=repr(exc),returned_steps=0)
 status['wall_seconds']=time.perf_counter()-started;return p,status
def qualify(model,x,t):
 q=x[0,np.linspace(0,255,17,dtype=int)];rows=[]
 for dim in [2,3]:
  ctx=x[:,:,:dim];tt=np.broadcast_to(t[None,:,None],(*ctx.shape[:2],1)).copy();vf=make_fim_vf_fn(model,ctx,tt,'cpu');a=vf(q[:,:dim]);tensor=torch.tensor(ctx[None],dtype=torch.float32);times=torch.tensor(tt[None],dtype=torch.float32);mask=torch.ones((*tensor.shape[:-1],1),dtype=torch.bool)
  with torch.no_grad():b=model.get_prediction_for_eval(model.model_forward(tensor,times,torch.tensor(q[None,:,:dim],dtype=torch.float32),mask))[0,:,:dim].numpy()
  assert np.allclose(a,b,atol=2e-5,rtol=2e-5);rows.append(dict(dimension=dim,cached_direct_max_abs=float(abs(a-b).max())))
 local=make_fim_vf_fn(model,x[:,:,:2],np.broadcast_to(t[None,:,None],(len(x),len(t),1)).copy(),'cpu')(q[:,:2]);padded=x.copy();padded[:,:,2]=0.;pq=q.copy();pq[:,2]=0.;zp=make_fim_vf_fn(model,padded,np.broadcast_to(t[None,:,None],(len(x),len(t),1)).copy(),'cpu')(pq)[:,:2];assert np.allclose(local,zp,atol=2e-5,rtol=2e-5)
 return dict(native_interfaces=rows,zero_padding_max_abs=float(abs(local-zp).max()),query_count=17)
def main():
 started=time.perf_counter();manifest=read(OUT/'DATA_MANIFEST82_V1.json');assert manifest['protocol_sha256']==sha(ROOT/'PROTOCOL82_V1.md');assets=old.assets();weight=next(a for a in assets if a['relative'].endswith('.safetensors'));assert old.check_source()==70;model=load_fim_ode_local(pathlib.Path(weight['path']).parent,'cpu');model.eval()
 for parameter in model.parameters():parameter.requires_grad_(False)
 with np.load(manifest['rows'][0]['path'],allow_pickle=False) as a:qualification=qualify(model,a['context'],a['time'])
 art=storage('artifact')/'closure_screen_v1';art.mkdir(exist_ok=False);dump(OUT/'OPENING82_V1.json',dict(utc=utc(),protocol_sha256=sha(ROOT/'PROTOCOL82_V1.md'),data_manifest_sha256=sha(OUT/'DATA_MANIFEST82_V1.json'),qualification=qualification,model_parameters=sum(p.numel() for p in model.parameters()),source_files_checked=70,weights_sha256=weight['sha256'],affinity=ordinary.AFFINITY,device='cpu',new_neural_training_fits=0));fields=[];rollouts=[];candidates=[];models=[]
 for item in manifest['rows']:
  sid=item['mechanism'];seed=item['seed'];regime=item['regime'];tag=f'm{sid}_seed{seed}_{regime}';assert sha(item['path'])==item['sha256']
  with np.load(item['path'],allow_pickle=False) as a:x=a['context'].copy();t=a['time'].copy();q=a['queries'].copy();truth=a['field_target'].copy()
  state('PROGRESS82_V1.json',dict(status='RUNNING',tag=tag,field_records=len(fields),rollout_records=len(rollouts),elapsed_seconds=time.perf_counter()-started));tt=np.broadcast_to(t[None,:,None],(8,256,1)).copy();vf2=make_fim_vf_fn(model,x[:,:,:2],tt,'cpu');vf3=make_fim_vf_fn(model,x,tt,'cpu');methods={'FIM_local2':lambda q:vf2(q)[:,0],'FIM_full3_diagnostic':lambda q:vf3(np.c_[q,np.zeros(len(q))])[:,0]}
  for kind in ['weak_polynomial3','weak_rff128']:
   begin=time.perf_counter();these=[]
   for ridge in [.001,.1,10.]:
    pars=fit(x[:,:192,:2],t[:192],kind,ridge);fn=field(pars);loss=[];solvers=[]
    for j in range(8):
     tv=t[191:];u=lambda now,j=j:np.interp(now,tv,x[j,191:,1]);pred,solver=solve_control(fn,u,x[j,191,0],tv);solvers.append(solver)
     if solver['success'] and pred.shape==(65,1):loss.append(float(np.mean((pred[:,0]-x[j,191:,0])**2)))
    score=float(np.mean(loss)) if len(loss)==8 else None;row=dict(tag=tag,kind=kind,ridge=ridge,validation_mse=score,successful_validation_paths=len(loss),solvers=solvers,parameters=pars);these.append(row);candidates.append(row)
   valid=[r for r in these if r['validation_mse'] is not None]
   if not valid:methods[kind]=None;models.append(dict(tag=tag,kind=kind,all_candidates_failed=True));continue
   chosen=min(valid,key=lambda r:(r['validation_mse'],r['ridge']));pars=fit(x[:,:,:2],t,kind,chosen['ridge']);pars['adaptation_seconds']=time.perf_counter()-begin;mp=art/f'{tag}_{kind}_model.json';dump(mp,pars);models.append(dict(tag=tag,kind=kind,path=str(mp),sha256=sha(mp)));methods[kind]=field(pars)
  for kind,fn in methods.items():
   prediction=fn(q) if fn is not None else np.empty(0);good=prediction.shape==truth.shape and np.isfinite(prediction).all();score=ordinary.metric(prediction[:,None] if good else None,truth[:,None]);path=art/f'{tag}_{kind}_field.npz';np.savez_compressed(path,queries=q,target=truth,prediction=prediction);row=dict(tag=tag,mechanism=sid,seed=seed,regime=regime,kind=kind,information='full_three_state_context_diagnostic' if kind=='FIM_full3_diagnostic' else 'local_two_state_context',**score,path=str(path),sha256=sha(path));fields.append(row);print(json.dumps(dict(type='field',**{key:row[key] for key in ['tag','kind','nrmse']})),flush=True)
   for j,(omega,y0) in enumerate(zip([.8,1.5],[-.6,.6])):
    grid=np.linspace(0,6,257);u=lambda now,omega=omega:.75*np.sin(omega*now)+.25*np.cos(.5*omega*now);ref=solve_ivp(lambda now,y:[target(np.array([y[0],u(now)]),item['parameters'])],(0,6),[y0],t_eval=grid,method='DOP853',rtol=1e-9,atol=1e-11);assert ref.success
    if fn is None:pred=np.empty((0,1));solver=dict(success=False,nfev=0,message='All context candidates failed',wall_seconds=0)
    else:pred,solver=solve_control(fn,u,y0,grid)
    rp=art/f'{tag}_{kind}_plan{j}.npz';np.savez_compressed(rp,time=grid,plan=u(grid),initial_state=np.array([y0]),target=ref.y.T,prediction=pred);row=dict(tag=tag,mechanism=sid,seed=seed,regime=regime,kind=kind,plan_index=j,omega=omega,initial_state=y0,**ordinary.metric(pred,ref.y.T),solver=solver,path=str(rp),sha256=sha(rp));rollouts.append(row)
  print(json.dumps(dict(type='task_closed',tag=tag,fields=len(fields),rollouts=len(rollouts),elapsed_seconds=time.perf_counter()-started)),flush=True)
 assert len(fields)==64 and len(rollouts)==128 and len(candidates)==96 and old.check_source()==70 and sha(weight['path'])==weight['sha256'];dump(OUT/'RESULTS82_V1.json',dict(utc=utc(),status='CLOSURE_SCREEN_CLOSED_UNVERIFIED',fields=fields,rollouts=rollouts,candidates=candidates,models=models,context_tasks=16,new_neural_training_fits=0,evaluation_is_frozen_development=True,weights_unchanged=True,wall_seconds=time.perf_counter()-started,accepted_final_directions=0,overall_goal='ACTIVE_UNACHIEVED'));state('PROGRESS82_V1.json',dict(status='CLOSURE_SCREEN_CLOSED_UNVERIFIED',fields=64,rollouts=128,wall_seconds=time.perf_counter()-started))
if __name__=='__main__':main()
