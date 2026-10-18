from common87_v1 import *
from ordinary87_v1 import features,FAMILIES
import numpy as np,time
NOISES=[.01,.03,.1,.3,1.]
def weak_design(context):
 phi=features(context,3);b,t,n,_=context.shape;rows=[];ys=[]
 for length in [9,17,33]:
  for start in range(0,n-length+1,4):
   s=np.linspace(0,1,length);span=(length-1)*.05;w=s*s*(1-s)**2;wp=2*s*(1-s)*(1-2*s)/span;q=np.ones(length);q[1:-1:2]=4;q[2:-1:2]=2;q*=.05/3;mass=np.sum(q*w)
   rows.append(np.einsum('l,btlf->btf',q*w,phi[:,:,start:start+length])/mass);ys.append(-np.einsum('l,btld->btd',q*wp,context[:,:,start:start+length])/mass)
 a=np.concatenate(rows,axis=1);y=np.concatenate(ys,axis=1);h=np.einsum('brf,ij->brifj',a,np.eye(2)).reshape(b,a.shape[1]*2,20)
 return h,y.reshape(b,-1)
def source_coefficients(arm):
 out=[]
 for fam in FAMILIES:
  # Family is a filename to enumerate supplied source files, never a predictor feature.
  with np.load(RAW/'population_v1'/f'train_{fam}_{arm}_v1.npz') as z:
   x=features(z['locations'],3);f=z['drift']
   out.extend([np.linalg.lstsq(xx,yy,rcond=1e-9)[0].reshape(-1) for xx,yy in zip(x,f)])
 return np.asarray(out)
def prior(coef,k):
 if k==1:labels=np.zeros(len(coef),int)
 else:
  mean=coef.mean(0);a=np.argmax(np.sum((coef-mean)**2,axis=1));b=np.argmax(np.sum((coef-coef[a])**2,axis=1));centers=coef[[a,b]].copy()
  for _ in range(40):
   labels=np.argmin(np.sum((coef[:,None,:]-centers[None,:,:])**2,axis=-1),axis=1)
   assert all(np.any(labels==j) for j in range(k));new=np.stack([coef[labels==j].mean(0) for j in range(k)])
   if np.max(abs(new-centers))<1e-10:break
   centers=new
 entries=[]
 for j in range(k):
  x=coef[labels==j];mu=x.mean(0);cov=np.cov(x,rowvar=False);w,v=np.linalg.eigh(cov);cov=(v*np.maximum(w,1e-6))@v.T;entries.append(dict(weight=float(len(x)/len(coef)),mean=mu,cov=cov))
 return entries
def posterior(entries,h,y,noise):
 b=len(h);means=[];evidence=[];hh=np.einsum('bri,brj->bij',h,h);n=y.shape[1]
 for entry in entries:
  mu=entry['mean'];cov=entry['cov'];precision=np.linalg.inv(cov);res=y-np.einsum('bri,i->br',h,mu);a=precision[None]+hh/noise**2;g=np.einsum('bri,br->bi',h,res)/noise**2;shift=np.linalg.solve(a,g[...,None])[...,0];means.append(mu[None]+shift)
  quad=np.sum(res*res,axis=1)/noise**2-np.sum(g*shift,axis=1);logdet=np.linalg.slogdet(cov)[1]+np.linalg.slogdet(a)[1]+n*np.log(noise**2);evidence.append(np.log(entry['weight'])-.5*(quad+logdet+n*np.log(2*np.pi)))
 e=np.stack(evidence,axis=1);e-=e.max(axis=1,keepdims=True);weights=np.exp(e);weights/=weights.sum(axis=1,keepdims=True);return np.einsum('bk,bki->bi',weights,np.stack(means,axis=1)).reshape(b,10,2)
def load_prior(path):
 d=np.load(path);return [dict(weight=float(d['weights'][i]),mean=d['means'][i],cov=d['covariances'][i]) for i in range(len(d['weights']))]
def main():
 assert not (O/'EVAL_OPENING87_V1.json').exists();start=time.perf_counter();folder=RAW/'empirical_prior_v1';folder.mkdir(exist_ok=False);val=[]
 for fam in FAMILIES:
  with np.load(RAW/'population_v1'/f'val_{fam}_unselected_v1.npz') as z:val.append({k:z[k] for k in z.files})
 context=np.concatenate([v['context'] for v in val]);loc=np.concatenate([v['locations'] for v in val]);truth=np.concatenate([v['drift'] for v in val]);std=np.maximum(np.std(context,axis=(1,2)),1e-6);h,y=weak_design(context);rows=[];selections=[]
 for arm in ['whole_filter','unselected']:
  coef=source_coefficients(arm);choices=[]
  for k in [1,2]:
   entries=prior(coef,k);path=folder/f'{arm}_components{k}_v1.npz'
   with path.open('xb') as f:np.savez_compressed(f,weights=np.array([e['weight'] for e in entries]),means=np.stack([e['mean'] for e in entries]),covariances=np.stack([e['cov'] for e in entries]),source_fitted_coefficients=coef)
   for noise in NOISES:
    post=posterior(entries,h,y,noise);est=np.einsum('bgf,bfd->bgd',features(loc,3),post);err=est-truth;score=float(np.mean(abs(err)/std[:,None,:])*.05);row=dict(arm=arm,components=k,weak_noise_std=noise,val_normalized_field_l1_proxy=score,val_native_normalized_field_l1=100*score,val_raw_field_mse=float(np.mean(err*err)),prior_path=str(path),prior_sha256=sha(path));rows.append(row);choices.append(row)
  selected=min(choices,key=lambda z:(z['val_normalized_field_l1_proxy'],z['components'],z['weak_noise_std']));selections.append(selected);log(dict(stage='empirical_prior_val',selected=selected,eval_opened=False))
 write(O/'EMPIRICAL_SELECTION87_V1.json',dict(utc=utc(),status='SAME_SUPERVISION_EMPIRICAL_PRIORS_SEALED_BEFORE_TEST',rows=rows,selections=selections,source_labels_used='32 TRAIN field queries per mechanism only, no parameter or family labels',source_mechanisms_per_prior=2048,whole_filter_and_resampled_prior_identical=True,noise_likelihood_approximate=True,eval_opened=False,addendum_sha256=sha(C/'STRONG_PRIOR_CONTROL87_V1.md'),script_sha256=sha(__file__),wall_seconds=time.perf_counter()-start))
if __name__=='__main__':main()
