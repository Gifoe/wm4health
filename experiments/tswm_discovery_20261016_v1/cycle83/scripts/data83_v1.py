import sys,pathlib
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common83_v1 import *
import numpy as np
from scipy.integrate import solve_ivp
def parents(family):
 if family=='fhn':return [p for i in range(6) for p in [[2*i,2*i+1,2*((i-1)%6)],[2*i+1,2*i]]]
 return [[i,(i-1)%7] for i in range(7)]
def true_field(x,family,setting):
 x=np.asarray(x);out=np.empty_like(x)
 if family=='fhn':
  g,base=[(.15,.3),(.4,.6)][setting]
  for i in range(6):
   v=x[...,2*i];w=x[...,2*i+1];prev=x[...,2*((i-1)%6)]
   out[...,2*i]=v-v**3/3-w+g*(prev-v)+base+.1*np.sin(i)
   out[...,2*i+1]=(.15+.02*i)*(v+.7-.8*w)
 else:
  alpha,beta=[(2.,1.),(4.,.8)][setting]
  for i in range(7):out[...,i]=alpha*(1+.05*np.sin(i))/(1+x[...,(i-1)%7]**3)-beta*x[...,i]
 return out
def initials(rng,family,n):
 if family=='hill':return rng.uniform(.5,2.,(n,7))
 x=np.empty((n,12));x[:,::2]=rng.normal(0,.7,(n,6));x[:,1::2]=rng.uniform(-.5,.5,(n,6));return x
def plan(now,initial):return initial+.5*np.sin(.8*now)
def reference(family,setting,y0,t,clamp=False,method='DOP853',rtol=1e-9,atol=1e-11):
 def rhs(now,y):
  q=y.copy()
  if clamp:q[0]=plan(now,y0[0])
  value=true_field(q,family,setting)
  if clamp:value[0]=.4*np.cos(.8*now)
  return value
 sol=solve_ivp(rhs,(float(t[0]),float(t[-1])),y0,t_eval=t,method=method,rtol=rtol,atol=atol);assert sol.success and len(sol.t)==len(t)
 p=sol.y.T
 if clamp:p[:,0]=plan(t,y0[0])
 return p
def main():
 root=storage('data')/'networks_v1';root.mkdir(exist_ok=False);rows=[];t=np.linspace(0,8,256)
 for family in ['fhn','hill']:
  for setting in range(2):
   for seed in [2026831,2026832]:
    rng=np.random.default_rng(seed);y0=initials(rng,family,4);clean=np.stack([reference(family,setting,z,t) for z in y0]);test_initials=initials(np.random.default_rng(seed+10000),family,2);noise=rng.normal(size=clean.shape);query_idx=np.random.default_rng(seed+20000).choice(1024,128,replace=False);queries=clean.reshape(1024,-1)[query_idx]
    for sigma in [0.,.03]:
     tag=f'{family}{setting}_s{seed}_noise{int(sigma*100):02d}';x=clean+sigma*clean.std((0,1))*noise;p=root/(tag+'.npz');np.savez_compressed(p,context=x,clean_context=clean,time=t,initial_states=y0,test_initial_states=test_initials,queries=queries,field_target=true_field(queries,family,setting));rows.append(dict(tag=tag,family=family,setting=setting,seed=seed,noise_sigma=sigma,parents=parents(family),path=str(p),sha256=sha(p),context_shape=list(x.shape)))
 dump(OUT/'DATA_MANIFEST83_V1.json',dict(utc=utc(),protocol_sha256=sha(ROOT/'PROTOCOL83_V1.md'),rows=rows));print(json.dumps(dict(status='DATA_FROZEN',tasks=len(rows),protocol_sha256=sha(ROOT/'PROTOCOL83_V1.md'))),flush=True)
if __name__=='__main__':main()
