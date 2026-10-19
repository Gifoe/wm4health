"""Exact scalar-input piecewise-affine analysis of unchanged LeakyReLU MLPs.

All parent coordinates are fixed; z is the final input coordinate. Integrate
phi(g(z))*abs(g'(z)) over the entire real line, not a truncated plotting grid.
This implements ordinary change-of-variables accounting, no novelty claimed.
"""
import numpy as np
from scipy.special import ndtr
def segments(net,parents,max_regions=200000):
 import torch.nn as nn
 parents=np.asarray(parents,dtype=np.float64);a=np.zeros((1,len(parents)+1));a[0,-1]=1.;b=np.r_[parents,0.][None];left=np.array([-np.inf]);right=np.array([np.inf])
 for layer in net.net:
  if isinstance(layer,nn.Linear):
   W=layer.weight.detach().cpu().double().numpy();bias=layer.bias.detach().cpu().double().numpy() if layer.bias is not None else 0.;a=a@W.T;b=b@W.T+bias
  elif isinstance(layer,nn.LeakyReLU):
   la=[];ra=[];aa=[];bb=[]
   for i in range(len(left)):
    roots=np.divide(-b[i],a[i],out=np.full_like(b[i],np.nan),where=a[i]!=0)
    roots=np.unique(roots[np.isfinite(roots)&(roots>left[i])&(roots<right[i])]);edges=np.r_[left[i],roots,right[i]]
    lo,hi=edges[:-1],edges[1:];mid=np.empty(len(lo));both=np.isfinite(lo)&np.isfinite(hi);mid[both]=(lo[both]+hi[both])/2
    mid[~np.isfinite(lo)&np.isfinite(hi)]=hi[~np.isfinite(lo)&np.isfinite(hi)]-np.maximum(1.,np.abs(hi[~np.isfinite(lo)&np.isfinite(hi)]))
    mid[np.isfinite(lo)&~np.isfinite(hi)]=lo[np.isfinite(lo)&~np.isfinite(hi)]+np.maximum(1.,np.abs(lo[np.isfinite(lo)&~np.isfinite(hi)]))
    mid[~np.isfinite(lo)&~np.isfinite(hi)]=0.
    factors=np.where(mid[:,None]*a[i]+b[i]>=0.,1.,layer.negative_slope)
    la.append(lo);ra.append(hi);aa.append(factors*a[i]);bb.append(factors*b[i])
   left=np.concatenate(la);right=np.concatenate(ra);a=np.concatenate(aa);b=np.concatenate(bb)
   if len(left)>max_regions:raise RuntimeError(f'Region cap exceeded: {len(left)}')
  else:raise TypeError(type(layer))
 assert a.shape[1]==1
 return left,right,a[:,0],b[:,0]
def endpoint(slope,bias,x):
 out=np.empty_like(slope);finite=np.isfinite(x);out[finite]=slope[finite]*x[finite]+bias[finite]
 out[~finite]=np.sign(x[~finite])*np.sign(slope[~finite])*np.inf
 out[(~finite)&(slope==0)]=bias[(~finite)&(slope==0)]
 return out
def audit(net,parents):
 left,right,a,b=segments(net,parents);u=endpoint(a,b,left);v=endpoint(a,b,right)
 mass=float(np.abs(ndtr(v)-ndtr(u)).sum());image_low=np.minimum(u,v);image_high=np.maximum(u,v)
 lo=float(image_low.min());hi=float(image_high.max());reachable=float(ndtr(hi)-ndtr(lo))
 signs=np.sign(a[np.abs(a)>1e-16]);folds=int(np.sum(signs[1:]!=signs[:-1]))
 roots={str(e):int(np.sum((image_low<e)&(image_high>e)&(a!=0))) for e in [-3.,-1.,0.,1.,3.]}
 return {'regions':len(left),'integrated_surrogate_mass':mass,'absolute_mass_error':abs(mass-1.),'base_noise_mass_with_preimage':reachable,'derivative_sign_changes':folds,'root_counts_at_noise_values':roots,'scalar_conditioning_only':True}

def normalized_samples(net,parents,u):
 from scipy.special import ndtri
 left,right,a,b=segments(net,parents);lo=endpoint(a,b,left);hi=endpoint(a,b,right);pl=ndtr(lo);pr=ndtr(hi)
 weights=np.abs(pr-pl);total=weights.sum();cumulative=np.cumsum(weights)
 target=np.asarray(u)*total;idx=np.minimum(np.searchsorted(cumulative,target,side='right'),len(weights)-1);base=np.r_[0.,cumulative[:-1]]
 probability=pl[idx]+np.sign(a[idx])*(target-base[idx]);probability=np.clip(probability,np.finfo(float).eps,1-np.finfo(float).eps)
 return (ndtri(probability)-b[idx])/a[idx]
