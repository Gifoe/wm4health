"""Standard affine output-error identification and ordinary monotone simulation fit."""
import numpy as np,torch,time
from scipy.optimize import lsq_linear,minimize
torch.set_num_threads(2)
def basis(x,k):
 k=np.asarray(k);w=np.diff(k);v=np.clip(x[...,None]-k[:-1],0,w);v[...,0]+=np.minimum(x-k[0],0);v[...,-1]+=np.maximum(x-k[-1],0)
 return np.concatenate([np.ones(x.shape+(1,)),v],axis=-1)
def pack(coef,k,uh,xh,fu):
 n,H=fu.shape;dim=len(k);R=np.concatenate([basis(xh[:,-64:],k),np.zeros((n,H,dim))],axis=1);d=np.zeros((n,64+H));u=np.concatenate([uh[:,-64:],fu],axis=1);a=coef[66:];b=coef[2:66]
 for j in range(H):
  t=64+j;ix=t-np.arange(1,65);R[:,t,:]=np.einsum('nlp,l->np',R[:,ix,:],a,optimize=True);d[:,t]=coef[0]+coef[1]*u[:,t]+u[:,ix]@b+d[:,ix]@a
 return np.ascontiguousarray(R[:,64:,:]),np.ascontiguousarray(d[:,64:])
def inverse_map(x,k,beta):return basis(x,k)@beta
def forward_map(y,k,beta):
 k=np.asarray(k);v=np.r_[beta[0],beta[0]+np.cumsum(np.diff(k)*beta[1:])];ix=np.clip(np.searchsorted(v,y,side='right')-1,0,len(k)-2)
 return k[ix]+(y-v[ix])/beta[1+ix]
def prediction(R,d,k,beta):return forward_map(R@beta+d,k,beta)
def torch_prediction(R,d,k,beta):
 v=torch.cat([beta[:1],beta[0]+torch.cumsum(torch.diff(k)*beta[1:],dim=0)]);latent=R@beta+d;ix=(torch.searchsorted(v,latent.contiguous(),right=True)-1).clamp(0,len(k)-2)
 return k[ix]+(latent-v[ix])/beta[ix+1]
def affine_fit(coef,uh,xh,fu,target):
 # A two-knot inverse spans the same affine family, without using paired canonical outputs.
 k=np.array([float(xh.min()),float(xh.max())]);assert k[1]>k[0]
 # Derive responses using the simple basis[1,x], which is independent of knot position.
 R,d=pack(coef,k,uh,xh,fu);R1=R[...,0];Rx=R[...,1]+k[0]*R1
 design=np.column_stack([d.ravel(),(R1-1).ravel()]);rhs=(target-Rx).ravel();solution=lsq_linear(design,rhs,bounds=([.02,-np.inf],[50.,np.inf]),method='bvls',tol=1e-12,max_iter=500);assert solution.success
 q,r=solution.x;a=1/q;b=r/q;beta=np.array([a*k[0]+b,a]);p=prediction(R,d,k,beta);direct=Rx+q*d+r*(R1-1);assert abs(p-direct).max()<1e-9
 residual=design@solution.x-rhs;grad=design.T@residual;free=solution.active_mask==0;kkt=float(abs(grad[free]).max()) if free.any() else 0.
 return dict(kind='affine_simulation',knots=k.tolist(),beta=beta.tolist(),q=float(q),r=float(r),training_simulation_mse=float(np.mean((p-target)**2)),design_condition_number=float(np.linalg.cond(design)),design_rank=int(np.linalg.matrix_rank(design)),optimizer_status=int(solution.status),active_bounds=int(np.count_nonzero(solution.active_mask)),normal_equation_free_coordinate_max_abs=kkt,affine_derivation_max_abs=float(abs(p-direct).max()))
def qualify(coef,k,beta,uh,xh,fu):
 # Calibration-only numerical qualification, not a DEV score.
 R,d=pack(coef,k,uh[:3],xh[:3],fu[:3]);pred=prediction(R,d,k,beta);rows=[]
 for i in range(len(pred)):
  us=list(uh[i]);ys=list(inverse_map(xh[i],k,beta));output=[]
  for v in fu[i]:
   z=coef[0]+coef[1]*v
   for j in range(1,65):z+=coef[1+j]*us[-j]+coef[65+j]*ys[-j]
   us.append(float(v));ys.append(float(z));output.append(z)
  p=forward_map(np.asarray(output),k,beta);rows.append(float(abs(p-pred[i]).max()))
 assert max(rows)<1e-8
 # Use strictly interior positive slopes so central differences do not encounter bounds.
 tb=np.r_[float(np.mean(xh[:3])),np.linspace(.7,1.3,len(k)-1)];tr=torch.tensor(R);td=torch.tensor(d);tk=torch.tensor(k);param=torch.tensor(tb,requires_grad=True);reference=torch_prediction(tr,td,tk,param);probe=np.sin(np.arange(reference.numel())).reshape(reference.shape);loss=(reference*torch.tensor(probe)).mean();loss.backward();automatic=param.grad.numpy().copy();finite=[];eps=1e-6
 for j in range(len(tb)):
  plus=tb.copy();minus=tb.copy();plus[j]+=eps;minus[j]-=eps;finite.append(float(((prediction(R,d,k,plus)-prediction(R,d,k,minus))*probe).mean()/(2*eps)))
 finite=np.asarray(finite);gradient_error=float(abs(automatic-finite).max());numpy_error=float(abs(reference.detach().numpy()-prediction(R,d,k,tb)).max());assert numpy_error<1e-9 and gradient_error<1e-5
 return dict(response_decomposition_vs_scalar_max_abs=max(rows),torch_numpy_forward_max_abs=numpy_error,autodiff_finite_difference_max_abs=gradient_error,gradient_coordinates=len(tb),future_outputs_used_in_response_decomposition=False)
def monotone_fit(coef,k,train,valid,initializers,scale,progress):
 uh,xh,fu,target=train;R,d=pack(coef,k,uh,xh,fu);vr,vd=pack(coef,k,*valid[:3]);tr=torch.tensor(R);td=torch.tensor(d);tk=torch.tensor(k);ty=torch.tensor(target);bounds=[(None,None)]+[(.02,50.)]*(len(k)-1);out=[];best=None
 for name,beta0 in initializers:
  beta0=np.asarray(beta0,dtype=np.float64);assert np.all(beta0[1:]>=.02) and np.all(beta0[1:]<=50)
  checkpoints=[];iteration=0;started=time.perf_counter();calls=0;last=None
  def objective(beta):
   nonlocal calls,last
   calls+=1;param=torch.tensor(beta,requires_grad=True);p=torch_prediction(tr,td,tk,param);loss=torch.mean(((p-ty)/scale)**2);loss.backward();value=float(loss.detach());gradient=param.grad.numpy().copy();assert np.isfinite(value) and np.isfinite(gradient).all();last=(value,gradient);return value,gradient
  def checkpoint(beta,stage):
   nonlocal best
   p=prediction(vr,vd,k,beta);mse=float(np.mean(((p-valid[3])/scale)**2));finite=bool(np.isfinite(p).all() and np.isfinite(mse));row=dict(initializer=name,iteration=iteration,stage=stage,internal_validation_mse=mse if finite else None,objective_evaluations=calls);checkpoints.append(row);progress(dict(**row,elapsed_seconds=time.perf_counter()-started))
   if finite and (best is None or mse<best['score']):best=dict(score=mse,beta=np.array(beta).copy(),initializer=name,iteration=iteration,stage=stage)
  def callback(beta):
   nonlocal iteration
   iteration+=1
   if iteration%50==0:checkpoint(beta,'periodic')
  checkpoint(beta0,'initial');sol=minimize(objective,beta0,method='L-BFGS-B',jac=True,bounds=bounds,callback=callback,options=dict(maxiter=300,maxls=40,gtol=1e-9,ftol=1e-12));checkpoint(sol.x,'terminal')
  value,gradient=objective(sol.x);projected=gradient.copy();projected[1:]=np.where((sol.x[1:]<=.02+1e-10)&(gradient[1:]>0),0,projected[1:]);projected[1:]=np.where((sol.x[1:]>=50.-1e-10)&(gradient[1:]<0),0,projected[1:])
  out.append(dict(initializer=name,initial_beta=beta0.tolist(),terminal_beta=sol.x.tolist(),optimizer_success=bool(sol.success),optimizer_status=int(sol.status),optimizer_message=str(sol.message),iterations=int(sol.nit),objective_evaluations=calls,terminal_normalized_training_mse=value,projected_gradient_max_abs=float(abs(projected).max()),checkpoints=checkpoints,elapsed_seconds=time.perf_counter()-started))
 assert best is not None
 model=dict(kind='monotone_simulation',knots=np.asarray(k).tolist(),beta=best['beta'].tolist(),selected_initializer=best['initializer'],selected_iteration=best['iteration'],selected_stage=best['stage'],internal_validation_normalized_mse=best['score'])
 return model,out
