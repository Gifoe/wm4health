"""CPU-only known-process reference qualification on VAL histories, not EVAL."""
import pathlib,sys,json,hashlib,math,datetime,os,time,traceback
R=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(R.parent/'cycle29/scripts'));import runtime_profile
import numpy as np
O=R/'outputs/reference_validation_v1';assert not O.exists();O.mkdir(parents=True)
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def dump(p,v):pathlib.Path(p).write_text(json.dumps(v,indent=2),encoding='utf-8')
H=8;FUT=16;PIXELS=64;COUNTS=[16384,65536];N_FUTURE=16384
state={'status':'running','pid':os.getpid(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'eval_opened':False};dump(O/'status.json',state)
def evolve(z,case,A):
 if case=='gaussian':return z@A.T
 return np.column_stack((.82*z[:,0]+.25*np.tanh(2*z[:,0])+.05*z[:,1],.65*z[:,1]+.2*np.sin(z[:,0]),.85*z[:,2]-.18*z[:,3]+.10*np.tanh(z[:,0]*z[:,1]),.18*z[:,2]+.85*z[:,3]))
def projected_history(history,truth,normalization):
 physical=history*normalization['scale']+np.asarray(normalization['mean']);return physical@np.asarray(truth['basis'])/2.5
def gaussian_posterior(history,truth,normalization):
 A=np.asarray(truth['A']);Q=np.diag(np.asarray(truth['process_std'])**2);cov=np.eye(4)
 for _ in range(100):cov=A@cov@A.T+Q
 observation=projected_history(history,truth,normalization);mean=np.zeros(4);D=np.eye(4)*(truth['observation_std']/2.5)**2
 for t,x in enumerate(observation):
  if t:mean=A@mean;cov=A@cov@A.T+Q
  gain=np.linalg.solve(cov+D,cov).T;mean+=gain@(x-mean);left=np.eye(4)-gain;cov=left@cov@left.T+gain@D@gain.T
 return mean,(cov+cov.T)/2
def particle_posterior(history,truth,normalization,n,draw_seed):
 rng=np.random.default_rng(draw_seed);A=np.asarray(truth['A']);q=np.asarray(truth['process_std']);z=rng.normal(size=(n,4));case=truth['case']
 for _ in range(100):z=evolve(z,case,A)+rng.normal(size=(n,4))*q
 observations=projected_history(history,truth,normalization);logw=np.zeros(n);ess=[]
 for t,y in enumerate(observations):
  if t:z=evolve(z,case,A)+rng.normal(size=(n,4))*q
  logw-=.5*np.sum(((z-y)/(truth['observation_std']/2.5))**2,axis=1);logw-=logw.max();w=np.exp(logw);w/=w.sum();effective=1/np.square(w).sum();ess.append(float(effective))
  if effective<n*.5 and t<H-1:
   points=(rng.random()+np.arange(n))/n;cdf=np.cumsum(w);cdf[-1]=1.;ix=np.searchsorted(cdf,points);z=z[ix];logw=np.zeros(n)
  else:logw=np.log(np.maximum(w,1e-300))
 return z,w,ess
def future_functions(truth,normalization,spec,n,draw_seed,posterior):
 rng=np.random.default_rng(draw_seed);A=np.asarray(truth['A']);q=np.asarray(truth['process_std']);C=np.asarray(truth['C']);mu=np.asarray(normalization['mean']);sigma=normalization['scale'];blocks=[]
 for start in range(0,n,2048):
  size=min(2048,n-start)
  if len(posterior)==2 and np.asarray(posterior[0]).ndim==1:
   mean,cov=posterior;z=mean+rng.normal(size=(size,4))@np.linalg.cholesky(cov).T
  else:
   particles,weights,_=posterior;z=particles[rng.choice(len(particles),size=size,p=weights)].copy()
  spatial=[];rough=np.zeros(size);occupancy=np.zeros(size);previous=None
  for t in range(FUT):
   z=evolve(z,truth['case'],A)+rng.normal(size=(size,4))*q;field=(z@C.T+rng.normal(0,truth['observation_std'],size=(size,PIXELS))-mu)/sigma
   spatial.append(field.mean(1));occupancy+=(field>spec['point_threshold']).mean(1)
   if previous is not None:rough+=np.square(field-previous).mean(1)
   previous=field
  spatial=np.stack(spatial,1);maximum=spatial.max(1);tau=spec['smoothmax_tau'];smooth=maximum+tau*np.log(np.exp((spatial-maximum[:,None])/tau).mean(1));blocks.append(np.column_stack((spatial.mean(1),smooth,rough/(FUT-1),occupancy/FUT,(maximum>spec['spatial_mean_threshold']).astype(float))))
 return np.concatenate(blocks)
def mean_function_oracle(mean,cov,truth,normalization):
 A=np.asarray(truth['A']);Q=np.diag(np.asarray(truth['process_std'])**2);powers=[np.eye(4)]
 for _ in range(FUT):powers.append(powers[-1]@A)
 left=sum(powers[1:])/FUT;avg_cov=left@cov@left.T
 for r in range(1,FUT+1):
  sensitivity=sum(powers[:FUT-r+1])/FUT;avg_cov+=sensitivity@Q@sensitivity.T
 loading=np.asarray(truth['C']).mean(0)/normalization['scale'];expected=float(loading@(left@mean)-np.asarray(normalization['mean']).mean()/normalization['scale']);variance=float(loading@avg_cov@loading+truth['observation_std']**2/(FUT*PIXELS*normalization['scale']**2))
 return expected,variance
try:
 records=[];checks=[]
 for case in ['gaussian','nonlinear']:
  folder=R/'data'/case;val=np.load(folder/'val.npz')['observations'];truth=read(folder/'truth_model.json');normalization=read(folder/'normalization.json');spec=read(folder/'functional_spec.json');scales=np.asarray(spec['scales'])
  for context in range(4):
   history=val[context,:H];local=[];gaussian_reference=None
   if case=='gaussian':
    posterior=gaussian_posterior(history,truth,normalization);functions=future_functions(truth,normalization,spec,N_FUTURE*4,202644500+context,posterior);mean,se=functions.mean(0),functions.std(0,ddof=1)/np.sqrt(len(functions));analytic_mean,analytic_variance=mean_function_oracle(*posterior,truth,normalization);gaussian_reference=(mean,se)
    np.savez_compressed(O/f'{case}_ctx{context}_exact.npz',functions=functions,posterior_mean=posterior[0],posterior_covariance=posterior[1]);records.append({'case':case,'context':context,'kind':'exact_Kalman_conditional_MC','future_draws':len(functions),'mean':mean.tolist(),'mc_se':se.tolist(),'analytic_mean_function':analytic_mean,'analytic_mean_function_variance':analytic_variance,'analytic_mean_zscore':float(abs(mean[0]-analytic_mean)/max(se[0],1e-12))})
   for n in COUNTS:
    for rep in range(2):
     rngseed=202644600+(case=='nonlinear')*100+context*10+rep;posterior=particle_posterior(history,truth,normalization,n,rngseed);functions=future_functions(truth,normalization,spec,N_FUTURE,rngseed+10000,posterior);mean,se=functions.mean(0),functions.std(0,ddof=1)/np.sqrt(len(functions));item={'case':case,'context':context,'kind':'bootstrap_particle_filter','particles':n,'rep':rep,'particle_seed':rngseed,'future_seed':rngseed+10000,'future_draws':N_FUTURE,'history_ess':posterior[2],'mean':mean.tolist(),'conditional_mc_se':se.tolist()};records.append(item);local.append(item);np.savez_compressed(O/f'{case}_ctx{context}_n{n}_r{rep}.npz',functions=functions);print(json.dumps({'case':case,'context':context,'particles':n,'rep':rep,'min_history_ess':min(posterior[2])}),flush=True)
   low=[x for x in local if x['particles']==COUNTS[0]];high=[x for x in local if x['particles']==COUNTS[1]];low_mean=np.mean([x['mean'] for x in low],0);high_mean=np.mean([x['mean'] for x in high],0);se_mean=np.sqrt(sum(np.asarray(x['conditional_mc_se'])**2 for x in local))/2;difference=np.abs(low_mean-high_mean);tolerance=np.maximum(5*se_mean,.02*scales);passed=difference<=tolerance
   comparison={'case':case,'context':context,'scaled_count_difference':(difference/scales).tolist(),'scaled_tolerance':(tolerance/scales).tolist(),'count_convergence_pass':passed.tolist()}
   if gaussian_reference is not None:
    exact_mean,exact_se=gaussian_reference;high_se=np.sqrt(sum(np.asarray(x['conditional_mc_se'])**2 for x in high))/2;delta=np.abs(high_mean-exact_mean);limit=np.maximum(5*np.sqrt(high_se**2+exact_se**2),.02*scales);comparison.update(scaled_PF_vs_exact_difference=(delta/scales).tolist(),PF_vs_exact_pass=(delta<=limit).tolist())
   checks.append(comparison)
 summary={'status':'REFERENCE_READY' if all(all(x['count_convergence_pass']) and all(x.get('PF_vs_exact_pass',[True])) for x in checks) else 'REFERENCE_UNCERTAINTY_UNRESOLVED','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'eval_opened':False,'cases':2,'VAL_histories_per_case':4,'particle_counts':COUNTS,'particle_repetitions':2,'future_draws_per_particle_run':N_FUTURE,'records':records,'convergence_checks':checks,'source_sha256':sha(pathlib.Path(__file__)),'protocol_sha256':sha(R/'PROTOCOL.md'),'learners_or_selection_not_changed':True};dump(O/'reference_validation.json',summary);state.update(status='completed',result_status=summary['status'],completed_utc=summary['utc']);dump(O/'status.json',state);print(json.dumps({k:v for k,v in summary.items() if k not in ['records','convergence_checks']}),flush=True)
except Exception:
 state.update(status='failed',traceback=traceback.format_exc());dump(O/'status.json',state);raise
