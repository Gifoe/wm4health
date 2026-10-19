"""Oracle proper-score comparison across endogenous representations, no fitting."""
import runtime_profile
import contextlib,datetime,json,pathlib,sys,time,traceback
import numpy as np
from scipy.special import logsumexp
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/score_folding';OUT.mkdir(parents=True,exist_ok=True)
class Tee:
 def __init__(self,*files):self.files=files
 def write(self,s):
  for f in self.files:f.write(s);f.flush()
 def flush(self):
  for f in self.files:f.flush()
def fold(x):
 r=np.linalg.norm(x,axis=-1,keepdims=True)
 return np.stack([x[...,0]**2-x[...,1]**2,2*x[...,0]*x[...,1]],axis=-1)/np.maximum(r,1e-150)
def normalized(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-150)
def distance(x,y):return np.sqrt(np.mean(np.sum((x-y)**2,axis=-1),axis=-1))
def summarize(x):return {'mean':float(np.mean(x)),'iid_context_standard_error':float(np.std(x,ddof=1)/np.sqrt(len(x))),'n_contexts':len(x)}
def normal_log(y,mu,var):return -np.log(2*np.pi*var)-np.sum((y-mu)**2,axis=-1)/(2*var)
def mixture_log(y,mu,var,p):
 if p==0:return normal_log(y,mu,var)
 return logsumexp(np.stack([normal_log(y,mu,var)+np.log1p(-p),normal_log(y,-mu,var)+np.log(p)]),axis=0)
def paths(x,e,u,p,rot,rho):
 out=[];state=x.copy()
 for t in range(e.shape[1]):
  state=(state@rot.T)*rho*np.where(u[:,t,None]<p,-1.,1.)+np.sqrt(1-rho**2)*e[:,t]
  out.append(state.copy())
 return np.stack(out,axis=1)
def main():
 assert not (OUT/'results.json').exists(),'scientific run already complete; do not overwrite'
 cfg={'rho':.9,'rotation_radians':.3,'flip_probabilities':[0.,.1,.25,.4,.5],'horizons':[1,8],'seeds':[2021,2022,2023],'n_contexts':32768,'oracle_conditional_replicates':2,'scope':'Exact-kernel diagnostic, not a native method reproduction or new method','normalization':['Gaussian radial','unit circle'],'score':'oracle Bayes Energy risk, RMS time-block distance','paired_grid':True}
 (OUT/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8');start=time.time();rows=[];density=[];moments=[]
 theta=cfg['rotation_radians'];rot=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]);rho=cfg['rho'];var=1-rho*rho;n=cfg['n_contexts'];T=max(cfg['horizons'])
 for seed in cfg['seeds']:
  rng=np.random.default_rng(seed);x=rng.normal(size=(n,2));e1=rng.normal(size=(n,T,2));e2=rng.normal(size=(n,T,2));u1=rng.random((n,T));u2=rng.random((n,T));s1=rng.choice([-1.,1.],size=(n,1));s2=rng.choice([-1.,1.],size=(n,1))
  np.savez_compressed(OUT/f'design_seed{seed}.npz',context=x,innovation1=e1,innovation2=e2,flip_uniform1=u1,flip_uniform2=u2,initial_sign1=s1,initial_sign2=s2)
  fx=fold(x);assert np.max(np.abs(fold(-x)-fx))<1e-14
  for name,data in [('identity',x),('fold',fx)]:moments.append({'seed':seed,'representation':name,'mean':data.mean(axis=0).tolist(),'covariance':np.cov(data.T).tolist(),'radius_preservation_max_error':float(np.max(np.abs(np.linalg.norm(x,axis=-1)-np.linalg.norm(data,axis=-1))))})
  for p in cfg['flip_probabilities']:
   y1=paths(x,e1,u1,p,rot,rho);y2=paths(x,e2,u2,p,rot,rho)
   fy1,fy2=fold(y1),fold(y2)
   # Paired coupling also proves the lumping contract: changing the initial
   # sign and future flip signs can only change the hidden sign fiber.
   q1=paths(x*s1,e1*s1[:,None,:],u1,p,rot,rho);q2=paths(x*s2,e2*s2[:,None,:],u2,p,rot,rho)
   assert np.max(np.abs(fold(q1)-fy1))<1e-12
   mu=x@rot.T*rho;lp=mixture_log(y1[:,0],mu,var,p);lf=mixture_log(y1[:,0],mu,var,.5)
   loss=-lf+lp
   density.append({'seed':seed,'p':p,'identity_conditional_NLL':summarize(-lp),'fold_conditional_NLL':summarize(-lf),'predictive_MI_lost_by_fold':summarize(loss),'analytic_raw_MSE_increase':2*rho*rho*(1-2*p)**2,'MC_raw_MSE_increase':summarize(np.sum(((1-2*p)*mu)**2,axis=-1))})
   for H in cfg['horizons']:
    for name,first,second,f_first,f_second in [('Gaussian radial',y1,y2,fy1,fy2),('unit circle',normalized(y1),normalized(y2),normalized(fy1),normalized(fy2))]:
     risk=.5*distance(first[:,:H],second[:,:H]);foldrisk=.5*distance(f_first[:,:H],f_second[:,:H]);delta=foldrisk-risk
     row={'seed':seed,'p':p,'horizon':H,'normalization':name,'identity_oracle_ES':summarize(risk),'fold_oracle_ES':summarize(foldrisk),'fold_minus_identity_ES':summarize(delta),'raw_folded_context_oracle_ES':summarize(.5*distance(q1[:,:H],q2[:,:H])) if name=='Gaussian radial' else None}
     rows.append(row)
   print(json.dumps({'seed':seed,'p':p,'one_step_MI_loss':float(np.mean(loss)),'raw_MSE_increase':2*rho*rho*(1-2*p)**2,'H1_energy_difference':rows[-4]['fold_minus_identity_ES']['mean'],'H8_energy_difference':rows[-2]['fold_minus_identity_ES']['mean']}),flush=True)
 result={'config':cfg,'score_rows':rows,'density_rows':density,'frame_moments':moments,'elapsed_seconds':time.time()-start,'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'interpretation':'Propriety for a fixed target does not compare different encoded targets. No full-method failure or novelty claim follows automatically. Existing likelihood/raw-space controls must be assessed.'}
 assert len(rows)==60 and len(density)==15
 (OUT/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8');print('COMPLETE',result['elapsed_seconds'],flush=True)
if __name__=='__main__':
 with (OUT/'stdout.log').open('a',encoding='utf-8') as stdout,(OUT/'stderr.log').open('a',encoding='utf-8') as stderr,contextlib.redirect_stdout(Tee(sys.stdout,stdout)),contextlib.redirect_stderr(Tee(sys.stderr,stderr)):
  try:main()
  except BaseException:traceback.print_exc();raise
