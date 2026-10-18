"""Generate separate observation-only fit arrays and scorer-only audit truth."""
from common61 import *
import numpy as np

def law(family):
 d=np.arange(1,33);p=[]
 for k in range(3):
  if family=='peaked':v=np.exp(-.5*((d-[8,13,20][k])/[1.3,1.9,2.5][k])**2)
  else:v=.45*np.exp(-.5*((d-[3,5,7][k])/1.1)**2)+.55*np.exp(-.5*((d-[17,22,27][k])/1.6)**2)
  p.append(v/v.sum())
 return {'duration':np.array(p),'initial':np.array([.2,.5,.3]),'transition':np.array([[0,.7,.3],[.4,0,.6],[.65,.35,0]]),'mean':np.array([[-1.,.4],[0.,-.8],[1.1,.7]]),'sd':np.array([[.5,.4],[.5,.4],[.5,.4]])}

def generate(family,n,seed,lengths):
 r=np.random.default_rng(seed);q=law(family);T=int(max(lengths));x=np.zeros((n,T,2));z=np.full((n,T),-1,dtype=np.int64);episode=np.full((n,T),-1,dtype=np.int64);age=np.zeros((n,T),dtype=np.int64)
 for b,L in enumerate(lengths):
  t=0;state=int(r.choice(3,p=q['initial']));e=0
  while t<L:
   duration=int(r.choice(np.arange(1,33),p=q['duration'][state]));end=min(int(L),t+duration);z[b,t:end]=state;episode[b,t:end]=e;age[b,t:end]=np.arange(1,end-t+1)
   x[b,t:end]=q['mean'][state]+q['sd'][state]*r.normal(size=(end-t,2));t=end;state=int(r.choice(3,p=q['transition'][state]));e+=1
 return x,z,episode,age

def main():
 assert read(OUT/'preflight_state61_v2.json')['status']=='NATIVE_COMPONENT_QUALIFICATION_READY'
 p=paths();rows={};truthrows={}
 for i,family in enumerate(['peaked','bimodal']):
  target=p['data_root']/family;target.mkdir(exist_ok=False);audit=p['artifact_root']/family;audit.mkdir(exist_ok=False);rows[family]={};truthrows[family]={}
  for j,(split,n) in enumerate([('train',96),('validation',32),('test',128)]):
   seed=61100+100*i+j;r=np.random.default_rng(seed+17);lengths=r.integers(80,129,size=n) if split=='train' else np.full(n,144);x,z,e,a=generate(family,n,seed,lengths)
   file=target/(split+'.npz')
   with file.open('xb') as f:np.savez(f,X=x.astype('float32'),lengths=lengths)
   truth=audit/(split+'_audit_truth.npz')
   with truth.open('xb') as f:np.savez(f,X_float64=x,state=z,episode=e,age=a,lengths=lengths,**law(family))
   rows[family][split]={'file':str(file),'sha256':sha(file),'records':n,'shape':list(x.shape),'generation_seed':seed,'keys':['X','lengths']}
   truthrows[family][split]={'file':str(truth),'sha256':sha(truth),'scorer_only':True}
 dump(OUT/'DATA61_V1.json',{'utc':utc(),'status':'UNLABELED_OBSERVATION_ARRAYS_AND_SEPARATE_AUDIT_TRUTH_PREPARED','families':rows,'audit_truth':truthrows,'generator_sha256':sha(__file__),'protocol_sha256':sha(ROOT/'PROTOCOL61_V1.md'),'role_separation':'Training reads only X and lengths in train/validation arrays, never latent audit labels or law parameters. Generation/provenance construction reads all splits before model fitting; no test score or test-based selection.','real_domains':0,'overall_goal':'ACTIVE_UNACHIEVED','accepted_final_directions':0})
 print(json.dumps({'status':'C61_DATA_PREPARED','families':2,'train_records_per_family':96,'test_records_per_family':128}),flush=True)

if __name__=='__main__':main()
