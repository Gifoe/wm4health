"""Ordinary learners: APIs receive observed arrays only, no sensor generator/raw calibration y."""
import numpy as np
from scipy.optimize import lsq_linear
L=64
def design(u,y,t):return np.column_stack([np.ones(len(t)),u[t]]+[u[t-j] for j in range(1,L+1)]+[y[t-j] for j in range(1,L+1)])
def fit_arx(records):
 gram=np.zeros((130,130));rhs=np.zeros(130);n=0
 for u,x in records:
  for start in range(L,len(u),8192):
   t=np.arange(start,min(start+8192,len(u)));d=design(u,x,t);gram+=d.T@d;rhs+=d.T@x[t];n+=len(t)
 lam=.001*float(np.diag(gram)[1:].mean());p=np.eye(130)*lam;p[0,0]=0
 coef=np.linalg.solve(gram+p,rhs)
 return coef,dict(training_rows=n,ridge=lam,normal_equation_residual=float(abs((gram+p)@coef-rhs).max()))
def rollout(coef,uh,xh,fu):
 # Batch dimension is independent episodes. No future-output argument or global data access.
 n,H=fu.shape;u=np.concatenate([uh[:,-L:],fu],axis=1);y=np.concatenate([xh[:,-L:].copy(),np.zeros((n,H))],axis=1)
 with np.errstate(over='ignore',invalid='ignore'):
  for j in range(H):
   t=L+j;d=np.column_stack([np.ones(n),u[:,t],u[:,t-np.arange(1,L+1)],y[:,t-np.arange(1,L+1)]]);y[:,t]=d@coef
 return y[:,L:]
def basis(x,knots):
 x=np.asarray(x);k=np.asarray(knots);width=np.diff(k);v=np.clip(x[...,None]-k[:-1],0,width)
 v[...,0]+=np.minimum(x-k[0],0);v[...,-1]+=np.maximum(x-k[-1],0)
 return np.concatenate([np.ones(x.shape+(1,)),v],axis=-1)
def inv_map(x,model):return basis(x,model['knots'])@np.asarray(model['beta'])
def forward_map(y,model):
 k=np.asarray(model['knots']);b=np.asarray(model['beta']);v=inv_map(k,model);y=np.asarray(y)
 flat=y.ravel();r=np.interp(flat,v,k);lo=flat<v[0];hi=flat>v[-1];r[lo]=k[0]+(flat[lo]-v[0])/b[1];r[hi]=k[-1]+(flat[hi]-v[-1])/b[-1]
 return r.reshape(y.shape)
def fit_maps(coef,records,kind):
 values=np.concatenate([x for _,x in records]);knots=np.quantile(values,np.linspace(0,1,17)) if kind=='monotone' else np.array([values.min(),values.max()]);assert np.all(np.diff(knots)>0)
 ds=[];rs=[]
 for u,x in records:
  b=basis(x,knots);d=b[L:].copy()
  for j,a in enumerate(coef[66:],1):d-=a*b[L-j:len(x)-j]
  t=np.arange(L,len(u));driver=np.full(len(t),coef[0])+coef[1]*u[t]
  for j,c in enumerate(coef[2:66],1):driver+=c*u[t-j]
  ds.append(d);rs.append(driver)
 d=np.concatenate(ds);r=np.concatenate(rs);scale=float(np.square(d[:,1:]).sum()/(d.shape[1]-1));prior=np.zeros((d.shape[1]-1,d.shape[1]));prior[:,1:]=np.eye(d.shape[1]-1);out=[]
 for lam in [0.,1e-6,.001,.1]:
  weight=np.sqrt(lam*scale);a=np.concatenate([d,weight*prior]);b=np.r_[r,np.full(d.shape[1]-1,weight)]
  low=np.r_[-np.inf,np.full(d.shape[1]-1,.02)];high=np.r_[np.inf,np.full(d.shape[1]-1,50.)]
  solved=lsq_linear(a,b,bounds=(low,high),method='bvls',tol=1e-12,max_iter=500)
  assert solved.success and np.isfinite(solved.x).all(),solved.message
  out.append(dict(kind=kind,knots=knots.tolist(),beta=solved.x.tolist(),lambda_identity_slopes=lam,innovation_mse=float(np.mean((d@solved.x-r)**2)),optimizer_status=int(solved.status),optimality=float(solved.optimality),active_bounds=int(np.count_nonzero(solved.active_mask)),design_condition_number=float(np.linalg.cond(d)),training_rows=len(r)))
 return out
def adapter_rollout(coef,m,uh,xh,fu):return forward_map(rollout(coef,uh,inv_map(xh,m),fu),m)

def neural_refit(records,validation,seed,save_callback,progress_callback):
 import torch,copy,time
 from torch import nn
 torch.set_num_threads(2);torch.manual_seed(seed);rng=np.random.default_rng(seed)
 observed=np.concatenate([x for _,x in records]);mean=float(observed.mean());std=float(observed.std());assert std>0
 # Each valid start is explicitly inside one record, including recursive targets.
 us=[];xs=[];starts=[];offset=0
 for u,x in records:
  us.append(u);xs.append((x-mean)/std);starts.extend(range(offset+L,offset+len(u)-16));offset+=len(u)
 u=torch.tensor(np.concatenate(us),dtype=torch.float32);x=torch.tensor(np.concatenate(xs),dtype=torch.float32);starts=np.asarray(starts);lag=torch.arange(1,L+1)
 model=nn.Sequential(nn.Linear(129,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,1));optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
 vu,vx,vf,vy=validation;vys=(vy-mean)/std;vxs=(vx-mean)/std
 best=None;checkpoints=[];started=time.perf_counter()
 def forecast(uh,xh,fu):
  n,H=fu.shape;ut=torch.tensor(np.concatenate([uh[:,-L:],fu],axis=1),dtype=torch.float32);yt=torch.tensor(np.concatenate([xh[:,-L:],np.zeros((n,H))],axis=1),dtype=torch.float32)
  with torch.no_grad():
   for j in range(H):
    t=L+j;d=torch.cat([ut[:,t:t+1],ut[:,t-lag],yt[:,t-lag]],dim=1);yt[:,t]=yt[:,t-1]+model(d)[:,0]
  return yt[:,L:].numpy()
 for step in range(1,2501):
  batch=256 if step<=2000 else 64;t=torch.tensor(rng.choice(starts,size=batch,replace=True));optimizer.zero_grad(set_to_none=True)
  if step<=2000:
   d=torch.cat([u[t,None],u[t[:,None]-lag],x[t[:,None]-lag]],dim=1);pred=x[t-1]+model(d)[:,0];loss=torch.mean((pred-x[t])**2)
  else:
   hist=x[t[:,None]-torch.arange(L,0,-1)].clone();errors=[]
   for j in range(16):
    tt=t+j;d=torch.cat([u[tt,None],u[tt[:,None]-lag],hist[:,-L:].flip(1)],dim=1);pred=hist[:,-1]+model(d)[:,0];errors.append((pred-x[tt])**2);hist=torch.cat([hist,pred[:,None]],dim=1)
   loss=torch.stack(errors).mean()
  assert bool(torch.isfinite(loss)),('nonfinite_training_loss',seed,step)
  loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
  if step%500==0:
   p=forecast(vu,vxs,vf).astype(np.float64);finite=bool(np.isfinite(p).all());mse=float(np.mean((p-vys)**2)) if finite else None
   if mse is not None and not np.isfinite(mse):finite=False;mse=None
   row=dict(update=step,training_loss=float(loss),internal_validation_mse=mse,all_predictions_finite=finite);checkpoints.append(row);progress_callback(dict(seed=seed,**row,elapsed_seconds=time.perf_counter()-started))
   if finite and (best is None or mse<best['mse']):best=dict(mse=mse,step=step,state=copy.deepcopy(model.state_dict()))
 assert best is not None,('all_internal_checkpoints_failed',seed)
 model.load_state_dict(best['state']);artifact=save_callback(dict(state_dict=best['state'],observed_mean=mean,observed_std=std,selected_update=best['step'],seed=seed))
 return model,dict(seed=seed,observed_mean=mean,observed_std=std,selected_update=best['step'],checkpoints=checkpoints,updates=2500,parameters=sum(p.numel() for p in model.parameters()),elapsed_seconds=time.perf_counter()-started,artifact=artifact)
def neural_rollout(model,info,uh,xh,fu):
 import torch
 n,H=fu.shape;mean=info['observed_mean'];std=info['observed_std'];u=torch.tensor(np.concatenate([uh[:,-L:],fu],axis=1),dtype=torch.float32);y=torch.tensor(np.concatenate([(xh[:,-L:]-mean)/std,np.zeros((n,H))],axis=1),dtype=torch.float32);lag=torch.arange(1,L+1)
 with torch.no_grad():
  for j in range(H):
   t=L+j;d=torch.cat([u[:,t:t+1],u[:,t-lag],y[:,t-lag]],dim=1);y[:,t]=y[:,t-1]+model(d)[:,0]
 return y[:,L:].numpy().astype(np.float64)*std+mean
