"""Frozen native weights: retained RPM normalization versus native inference.

Known-observer Bayesian filtering is an attribution oracle, not a learned-model
baseline. All empirical mixture components come from training data only.
"""
from train_native_alarm import ROOT,construct
import argparse,hashlib,json,pathlib,pickle,time
import numpy as onp
from scipy.special import logsumexp,ndtr
from scipy.stats import norm
from threadpoolctl import threadpool_limits
import jax
import jax.numpy as np
from rp_ssm.datasets import TrainData

def factor_arrays(energy,params,obs):
 recparams=params[1:]
 fn=jax.jit(lambda x:energy.model.get_factors(recparams,(x,)))
 f=fn(np.array(obs)).params
 return onp.asarray(f['p'])[0,...,0,0],onp.asarray(f['pwm'])[0,...,0]

def normal_loglik(p,b,grid):
 # full recognition = normalized prior * positive site
 p=p+1.
 return .5*onp.log(p[:,None]/(2*onp.pi))-.5*(p[:,None]*grid[None,:]**2-2*b[:,None]*grid[None,:]+b[:,None]**2/p[:,None])

def mixture_logs(train_p,train_b,grid):
 return onp.stack([logsumexp(normal_loglik(train_p[:,t],train_b[:,t],grid),axis=0)-onp.log(len(train_p)) for t in range(train_p.shape[1])])

def filter_grid(plan,kind,arrays,grid,prefixes,query_factors=None,logF=None):
 rho=plan['rho'];step=grid[1]-grid[0];transition=norm.pdf(grid[None,:],loc=rho*grid[:,None],scale=onp.sqrt(1-rho*rho))*step
 N,T=arrays['obs'].shape[:2];weights=onp.tile(norm.pdf(grid),(N,1));weights/=weights.sum(axis=1,keepdims=True);outputs={}
 p,b=query_factors if query_factors is not None else (None,None)
 for t in range(T):
  if t:weights=weights@transition
  if p is not None:ll=normal_loglik(p[:,t],b[:,t],grid)-logF[t]
  else:
   x=arrays['obs'][:,t,0];mask=arrays['mask'][:,t];alarm=arrays['alarm'][:,t]
   ll=-.5*((x[:,None]-grid[None,:])/plan['anchor_noise_sd'])**2*mask[:,None]
   if kind=='state_dependent_alarm':
    logits=plan['alarm_intercept']+plan['alarm_quadratic_coefficient']*grid**2
    ll=ll+onp.where(alarm[:,None],-onp.logaddexp(0.,-logits),-onp.logaddexp(0.,logits))
  weights=weights*onp.exp(ll-ll.max(axis=1,keepdims=True));total=weights.sum(axis=1,keepdims=True)
  assert onp.isfinite(total).all() and (total>0).all();weights/=total
  if t+1 in prefixes:outputs[t+1]=weights.copy()
 return outputs

def moments(weights,grid):
 m=weights@grid;v=weights@(grid*grid)-m*m
 assert onp.isfinite(v).all() and (v>0).all()
 return m,v

def native_moments(energy,params,obs,L):
 prior=energy.model.prior.update(params[0])
 def get(x):return energy.get_posterior(None,prior,params[1:],TrainData(obs=(x,)))[2].params
 post=jax.jit(get)(np.array(obs[:,:L]));return onp.asarray(post['means'])[:,-1,0],onp.asarray(post['covs'])[:,-1,0,0]

def readout(m,z):
 coefficient=onp.linalg.lstsq(onp.stack([m,onp.ones_like(m)],axis=1),z,rcond=None)[0]
 return float(coefficient[0]),float(coefficient[1])

def summarize(plan,arrays,L,m,v,weights,grid,affine=(1.,0.),raw_scores=None,method_label=''):
 C,bias=affine;physical_m=C*m+bias;physical_v=C*C*v;z=arrays['z'][:,L-1];alarm=arrays['alarm'][:,L-1];mask=arrays['mask'][:,L-1]
 subsets={'all':onp.ones(len(z),bool),'alarm_no_anchor':alarm & ~mask,'alarm_no_anchor_last_four':alarm & ~arrays['mask'][:,max(0,L-4):L].any(axis=1)}
 rows=[]
 for h in [0]+plan['forecast_horizons']:
  a=plan['rho']**h;q=1-a*a;predm=C*a*m+bias;predv=C*C*(a*a*v+q)
  expected_squared_error=(predm-a*z)**2+q
  expected_nll=.5*(onp.log(2*onp.pi*predv)+expected_squared_error/predv)
  radius=1.959963984540054*onp.sqrt(predv)
  if h==0:
   coverage=(onp.abs(z-predm)<=radius).astype(float);target_tail=(onp.abs(z)>2).astype(float)
  else:
   sd=onp.sqrt(q);coverage=ndtr((predm+radius-a*z)/sd)-ndtr((predm-radius-a*z)/sd)
   target_tail=ndtr((-2-a*z)/sd)+ndtr((a*z-2)/sd)
  if weights is None:pred_tail=ndtr((-2-predm)/onp.sqrt(predv))+ndtr((predm-2)/onp.sqrt(predv))
  elif h==0:pred_tail=weights@(onp.abs(C*grid+bias)>2)
  else:
   sd=abs(C)*onp.sqrt(q);gmean=C*a*grid+bias
   pred_tail=weights@(ndtr((-2-gmean)/sd)+ndtr((gmean-2)/sd))
  expected_brier=pred_tail*pred_tail-2*pred_tail*target_tail+target_tail
  if raw_scores is not None:
   for score,values in [('mse',expected_squared_error),('nll',expected_nll),('coverage',coverage),('tail_brier',expected_brier),('predicted_tail',pred_tail),('target_tail',target_tail)]:
    raw_scores[f'L{L}_H{h}_{method_label}_{score}']=values
  for subset,selected in subsets.items():
   if not selected.any():continue
   rows.append({'prefix':L,'horizon':h,'subset':subset,'n':int(selected.sum()),'mse':float(expected_squared_error[selected].mean()),
    'moment_gaussian_nll':float(expected_nll[selected].mean()),'moment_gaussian_95_coverage':float(coverage[selected].mean()),'mean_predicted_variance':float(predv[selected].mean()),
    'fraction_predicted_variance_above_stationary':float((predv[selected]>C*C+1e-4).mean()),'expected_tail_brier':float(expected_brier[selected].mean()),
    'mean_predicted_abs_state_gt2':float(pred_tail[selected].mean()),'mean_realized_or_noise_expected_abs_state_gt2':float(target_tail[selected].mean())})
 return rows

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--plan',default='native_learning_plan.json');args=parser.parse_args()
 pp=ROOT/'outputs'/args.plan;plan=json.loads(pp.read_text(encoding='utf-8-sig'));out=ROOT/'outputs'/plan.get('output_subdirectory','native_alarm');start=time.perf_counter()
 gconf=plan['exact_grid'];grid=onp.linspace(-gconf['extent'],gconf['extent'],gconf['points']);allsummary=[]
 for kind in plan['datasets']:
  with onp.load(out/(kind+'_data.npz')) as f:
   arrays={split:{k:f[split+'_'+k] for k in ['obs','z','mask','alarm']} for split in ['train','val','test']}
  prefixes=plan['prefix_lengths']
  oracle=filter_grid(plan,kind,arrays['test'],grid,prefixes)
  for seed in plan['training_seeds']:
   stem=kind+'_seed'+str(seed);result_path=out/(stem+'_evaluation.json')
   if result_path.exists():print('existing evaluation retained',stem,flush=True);continue
   ckpt=out/(stem+'.pkl');params=pickle.loads(ckpt.read_bytes());energy,conf=construct(plan,seed)
   train_p,train_b=factor_arrays(energy,params,arrays['train']['obs']);test_p,test_b=factor_arrays(energy,params,arrays['test']['obs']);logF=mixture_logs(train_p,train_b,grid)
   exact=filter_grid(plan,kind,arrays['test'],grid,prefixes,(test_p,test_b),logF)
   # Fixed training-only affine state readout. No hidden state is used in fitting
   # recognition weights, dynamics or selecting their checkpoints.
   cal={k:v[:1024] for k,v in arrays['train'].items()};cal_exact=filter_grid(plan,kind,cal,grid,[16],(train_p[:1024],train_b[:1024]),logF)[16]
   cal_native=native_moments(energy,params,cal['obs'],16)[0]
   C,bias=readout(cal_native,cal['z'][:,-1]);Cex,bex=readout(moments(cal_exact,grid)[0],cal['z'][:,-1]);assert abs(C)>1e-5
   rows=[];raw={};discrepancies=[]
   for L in prefixes:
    mn,vn=native_moments(energy,params,arrays['test']['obs'],L);me,ve=moments(exact[L],grid);mo,vo=moments(oracle[L],grid)
    for method,m,v,w,affine in [('native',mn,vn,None,(C,bias)),('exact_same_readout',me,ve,exact[L],(C,bias)),('exact_refit_readout',me,ve,exact[L],(Cex,bex)),('known_observer_oracle',mo,vo,oracle[L],(1.,0.))]:
     for row in summarize(plan,arrays['test'],L,m,v,w,grid,affine,raw,method):row['method']=method;rows.append(row)
    discrepancies.append({'prefix':L,'mean_abs_posterior_mean_difference':float(onp.abs(me-mn).mean()),'mean_native_variance':float(vn.mean()),'mean_exact_variance':float(ve.mean()),'exact_variance_gt_native_fraction':float((ve>vn).mean())})
    for name,val in [('native_mean',mn),('native_variance',vn),('exact_mean',me),('exact_variance',ve),('oracle_mean',mo),('oracle_variance',vo)]:raw[str(L)+'_'+name]=val
   # Grid/range check on a fixed subset, selected by index without test scores.
   finegrid=onp.linspace(-gconf['validation_extent'],gconf['validation_extent'],gconf['validation_points']);ncheck=gconf['validation_samples'];small={k:v[:ncheck] for k,v in arrays['test'].items()}
   fineF=mixture_logs(train_p,train_b,finegrid);fine=filter_grid(plan,kind,small,finegrid,prefixes,(test_p[:ncheck],test_b[:ncheck]),fineF)
   checks=[]
   for L in prefixes:
    mf,vf=moments(fine[L],finegrid);mc,vc=moments(exact[L][:ncheck],grid)
    checks.append({'prefix':L,'max_mean_difference':float(onp.abs(mf-mc).max()),'max_variance_difference':float(onp.abs(vf-vc).max())})
   assert max(v['max_mean_difference'] for v in checks)<1e-4
   assert max(v['max_variance_difference'] for v in checks)<1e-4
   raw_path=out/(stem+'_evaluation_arrays.npz');onp.savez_compressed(raw_path,**raw)
   result={'dataset':kind,'seed':seed,'readout_native':{'slope':C,'intercept':bias},'readout_exact_separate_control':{'slope':Cex,'intercept':bex},
    'metrics':rows,'inference_discrepancies':discrepancies,'grid_checks':checks,'plan_sha256':hashlib.sha256(pp.read_bytes()).hexdigest(),
    'checkpoint_sha256':hashlib.sha256(ckpt.read_bytes()).hexdigest(),'raw_arrays_sha256':hashlib.sha256(raw_path.read_bytes()).hexdigest(),
    'scope':'Frozen same learned model, unknown observation mapping; exact grid inference is an existing baseline. Known observer oracle uses privileged likelihood and must not be counted as a fair learned baseline.',
    'metric_notes':'NLL/95 intervals use moment-matched Gaussians for exact/oracle methods; tail probabilities retain full grid mixtures. Future metrics average analytically over fresh transition noise conditional on the actual present test state, avoiding Monte Carlo trajectory noise.',
    'runtime_seconds_since_script_start':time.perf_counter()-start}
   result_path.write_text(json.dumps(result,indent=2),encoding='utf-8');allsummary.append(result)
   print(json.dumps({'dataset':kind,'seed':seed,'readout_slope':C,'primary':[r for r in rows if r['prefix']==16 and r['horizon']==1 and r['subset']=='alarm_no_anchor'],'elapsed_seconds':time.perf_counter()-start}),flush=True)
 print('all evaluations complete',time.perf_counter()-start,flush=True)
if __name__=='__main__':
 with threadpool_limits(limits=4):main()
