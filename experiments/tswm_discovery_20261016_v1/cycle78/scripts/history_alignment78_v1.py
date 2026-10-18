"""Ordinary nuisance estimators; no dataset/generator/future-data imports."""
import numpy as np
from scipy.optimize import lsq_linear,least_squares
def inverse(x,m):
 k=np.asarray(m['knots']);b=np.asarray(m['beta']);w=np.diff(k);r=np.clip(x[...,None]-k[:-1],0,w);r[...,0]+=np.minimum(x-k[0],0);r[...,-1]+=np.maximum(x-k[-1],0)
 return b[0]+r@b[1:]
def forward_and_slope(y,m):
 k=np.asarray(m['knots']);b=np.asarray(m['beta']);v=np.r_[b[0],b[0]+np.cumsum(np.diff(k)*b[1:])];i=np.clip(np.searchsorted(v,y,side='right')-1,0,len(k)-2);s=1/b[i+1]
 return k[i]+(y-v[i])*s,s
def responses(coef,uh,zh,fu):
 n,H=fu.shape;u=np.concatenate([uh[:,-64:],fu],axis=1);rh=np.concatenate([zh[:,-64:].copy(),np.zeros((n,H))],axis=1);d=np.zeros((n,64+H));one=np.concatenate([np.ones((n,64)),np.zeros((n,H))],axis=1)
 for j in range(H):
  t=64+j;ix=t-np.arange(1,65);rh[:,t]=rh[:,ix]@coef[66:];one[:,t]=one[:,ix]@coef[66:];d[:,t]=coef[0]+coef[1]*u[:,t]+u[:,ix]@coef[2:66]+d[:,ix]@coef[66:]
 return rh[:,64:],d[:,64:],one[:,64:]-1
def innovation_problem(coef,m,uh,xh):
 z=inverse(xh,m);times=np.arange(64,128);v=z[:,times].copy();driver=np.full_like(v,coef[0])+coef[1]*uh[:,times]
 for j in range(1,65):v-=coef[65+j]*z[:,times-j];driver+=coef[1+j]*uh[:,times-j]
 return v,driver,float(1-coef[66:].sum())
def fit_history(coef,m,uh,xh,residual_scale,method):
 # These arguments contain only128 historical observations. State resets for every call.
 assert uh.shape==xh.shape and uh.shape[1]==128;parameters=[];metadata=[];lam=float(residual_scale)
 if method in ['offset_innovation_ridge','affine_innovation_ridge']:
  v,driver,D=innovation_problem(coef,m,uh,xh)
  for i in range(len(uh)):
   if method=='offset_innovation_ridge':
    q=1.;unbounded=-D*float(np.sum(v[i]-driver[i]))/(64*D*D+lam);r=float(np.clip(unbounded,-2,2));cost=float(np.sum((v[i]-q*driver[i]+r*D)**2)+lam*r*r);meta=dict(success=True,status=0,evaluations=1,cost=cost,optimality=None,active_bounds=int(r in [-2.,2.]))
   else:
    X=np.column_stack([driver[i],np.full(64,-D)]);a=np.concatenate([X,np.sqrt(lam)*np.eye(2)]);b=np.r_[v[i],np.sqrt(lam),0.];sol=lsq_linear(a,b,bounds=([.02,-2.],[50.,2.]),method='bvls',tol=1e-12,max_iter=500);q,r=sol.x;meta=dict(success=bool(sol.success),status=int(sol.status),evaluations=int(sol.nit),cost=float(np.sum((a@sol.x-b)**2)),optimality=float(sol.optimality),active_bounds=int(np.count_nonzero(sol.active_mask)))
   parameters.append([float(q),float(r)]);metadata.append(meta)
 else:
  assert method=='affine_past_simulation';rh,d,s=responses(coef,uh[:,:64],inverse(xh[:,:64],m),uh[:,64:]);normalizer=np.maximum(xh.std(axis=1),.001)
  for i in range(len(uh)):
   scale=float(normalizer[i])
   def residual(par):
    p,_=forward_and_slope(rh[i]+par[0]*d[i]+par[1]*s[i],m);return np.r_[(p-xh[i,64:])/scale,np.sqrt(.001)*(par-np.array([1.,0.]))]
   def jacobian(par):
    _,slope=forward_and_slope(rh[i]+par[0]*d[i]+par[1]*s[i],m);return np.concatenate([np.column_stack([slope*d[i],slope*s[i]])/scale,np.sqrt(.001)*np.eye(2)],axis=0)
   sol=least_squares(residual,[1.,0.],jac=jacobian,bounds=([.02,-2.],[50.,2.]),method='trf',max_nfev=100,ftol=1e-10,xtol=1e-10,gtol=1e-10);parameters.append(sol.x.tolist());metadata.append(dict(success=bool(sol.success),status=int(sol.status),evaluations=int(sol.nfev),cost=float(np.sum(residual(sol.x)**2)),optimality=float(sol.optimality),active_bounds=int(np.count_nonzero(sol.active_mask)),history_x_normalizer=scale))
 return np.asarray(parameters),metadata
def forecast(coef,m,uh,xh,fu,parameters):
 rh,d,s=responses(coef,uh,inverse(xh,m),fu);q=parameters[:,0,None];r=parameters[:,1,None];pred,_=forward_and_slope(rh+q*d+r*s,m);return pred
def qualify(coef,m,uh,xh):
 # Observed calibration-fit history, not DEV. No future outputs or canonical labels.
 rh,d,s=responses(coef,uh[:,:64],inverse(xh[:,:64],m),uh[:,64:]);par=np.array([.9,.1]);p,derivative=forward_and_slope(rh+par[0]*d+par[1]*s,m);eps=1e-6;jac=np.stack([derivative*d,derivative*s],axis=-1);diff=[]
 for j in [0,1]:
  plus=par.copy();minus=par.copy();plus[j]+=eps;minus[j]-=eps;pp,_=forward_and_slope(rh+plus[0]*d+plus[1]*s,m);pm,_=forward_and_slope(rh+minus[0]*d+minus[1]*s,m);diff.append(float(abs((pp-pm)/(2*eps)-jac[...,j]).max()))
 assert max(diff)<1e-5
 # Independently simulate the corrected inverse, then invert it using the original map.
 out=[]
 for i in range(len(uh)):
  us=list(uh[i,:64]);ys=list((inverse(xh[i,:64],m)+par[1])/par[0]);pred=[]
  for v in uh[i,64:]:
   z=coef[0]+coef[1]*v
   for lag in range(1,65):z+=coef[lag+1]*us[-lag]+coef[lag+65]*ys[-lag]
   us.append(v);ys.append(z);val,_=forward_and_slope(np.array([par[0]*z-par[1]]),m);pred.append(val[0])
  out.append(pred)
 error=float(abs(np.asarray(out)-p).max());assert error<1e-8
 return dict(analytic_jacobian_finite_difference_max_abs=max(diff),corrected_map_vs_response_max_abs=error,parameter_fit_has_no_future_arguments=True)
