from common87_v1 import *
from physics87_v1 import FAMILIES
import numpy as np,time
DEGREES=[1,2,3];RIDGES=[0.,1e-6,1e-4,.01,1.]
def powers(degree):return [(i,j) for total in range(degree+1) for i in range(total+1) for j in [total-i]]
def features(x,degree):return np.stack([x[...,0]**i*x[...,1]**j for i,j in powers(degree)],axis=-1)
def identify(context,degree,ridge):
 b,t,n,_=context.shape;phi=features(context,degree);rows=[];targets=[]
 for length in [9,17,33]:
  for start in range(0,n-length+1,4):
   s=np.linspace(0,1,length);span=(length-1)*.05;w=s*s*(1-s)**2;wp=2*s*(1-s)*(1-2*s)/span
   q=np.ones(length);q[1:-1:2]=4;q[2:-1:2]=2;q*=.05/3
   rows.append(np.einsum('l,btlf->btf',q*w,phi[:,:,start:start+length]));targets.append(-np.einsum('l,btld->btd',q*wp,context[:,:,start:start+length]))
 a=np.concatenate(rows,axis=1);y=np.concatenate(targets,axis=1);scale=np.maximum(np.sqrt(np.mean(a*a,axis=1)),1e-8);design=a/scale[:,None,:]
 coef=[]
 for k in range(b):
  if ridge==0:beta=np.linalg.lstsq(design[k],y[k],rcond=1e-8)[0]
  else:beta=np.linalg.solve(design[k].T@design[k]+ridge*np.eye(design.shape[-1]),design[k].T@y[k])
  coef.append(beta/scale[k,:,None])
 return np.stack(coef)
def predict(coef,x,degree):return np.einsum('b...f,bfd->b...d',features(x,degree),coef)
def main():
 start=time.perf_counter();rows=[];per_family=[]
 for fam in FAMILIES:
  with np.load(RAW/'population_v1'/f'val_{fam}_unselected_v1.npz') as z:d={k:z[k] for k in z.files}
  per_family.append(d)
 context=np.concatenate([d['context'] for d in per_family]);loc=np.concatenate([d['locations'] for d in per_family]);truth=np.concatenate([d['drift'] for d in per_family]);std=np.maximum(np.std(context,axis=(1,2)),1e-6)
 for degree in DEGREES:
  for ridge in RIDGES:
   coef=identify(context,degree,ridge);est=predict(coef,loc,degree);err=est-truth
   score=float(np.mean(abs(err)/std[:,None,:])*.05);row=dict(degree=degree,ridge=ridge,val_normalized_field_l1=score,val_raw_field_mse=float(np.mean(err*err)),finite=bool(np.isfinite(est).all()));rows.append(row)
 best=min(rows,key=lambda z:(z['val_normalized_field_l1'],z['degree'],z['ridge']));assert best['finite'];write(O/'ORDINARY_SELECTION87_V1.json',dict(utc=utc(),status='FIFTEEN_ORDINARY_CHOICES_SEALED_ON_VAL_ONLY',rows=rows,selected=best,eval_opened=False,details_sha256=sha(C/'EVALUATION_DETAILS87_V1.md'),source_sha256=sha(__file__),wall_seconds=time.perf_counter()-start));log(dict(stage='ordinary_val_selection',selected=best,eval_opened=False))
if __name__=='__main__':main()
