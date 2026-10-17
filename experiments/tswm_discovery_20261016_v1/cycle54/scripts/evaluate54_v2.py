"""Frozen-fit state/readout and causal rollout audit, with independent Kalman checks."""
from common54 import *
import pickle,time
from scipy.optimize import linear_sum_assignment
from rp_ssm.datasets import TrainData
from rp_ssm.training import Trainer
from rp_ssm.dists_utils import transitions_to_marginals
from dynamax.linear_gaussian_ssm import LinearGaussianSSM

def kalman(x,A,b,Q,C,d,R,m1,P1):
 """Independent NumPy batch Kalman filter and RTS smoother, no upstream inference."""
 N,T,D=x.shape;K=len(m1);m=np.broadcast_to(m1,(N,K)).copy();P=np.broadcast_to(P1,(N,K,K)).copy()
 fm=[];fp=[];pm=[];pp=[]
 for t in range(T):
  if t:m=m@A.T+b;P=A@P@A.T+Q
  pm.append(m.copy());pp.append(P.copy())
  Rt=R[:,t] if R.ndim==4 else R
  S=C@P@C.T+Rt;gain=P@C.T@np.linalg.inv(S)
  m=m+np.einsum('nkd,nd->nk',gain,x[:,t]-m@C.T-d)
  P=P-gain@S@gain.swapaxes(-1,-2);P=(P+P.swapaxes(-1,-2))/2
  fm.append(m.copy());fp.append(P.copy())
 fm=np.stack(fm,1);fp=np.stack(fp,1);pm=np.stack(pm,1);pp=np.stack(pp,1)
 sm=fm.copy();sp=fp.copy()
 for t in range(T-2,-1,-1):
  G=fp[:,t]@A.T@np.linalg.inv(pp[:,t+1])
  sm[:,t]=fm[:,t]+np.einsum('nkl,nl->nk',G,sm[:,t+1]-pm[:,t+1])
  sp[:,t]=fp[:,t]+G@(sp[:,t+1]-pp[:,t+1])@G.swapaxes(-1,-2)
 return fm,fp,sm,sp

def components(method,task,seed,params,x):
 K=case(task)['latent_dim'];reported_jitter=1e-5 if method=='rp' else 0.
 if method=='rp':
  energy,cfg=rp_model(task,seed);trainer=Trainer(energy,cfg);trainer.params=params
  factors=energy.model.get_factors(params[1:],(jnp.asarray(x),))
  precision=np.asarray(factors.params['p'])[0];pwm=np.asarray(factors.params['pwm'])[0]
  pseudo=np.linalg.solve(precision,pwm[...,None])[...,0];R=np.linalg.inv(precision)
  A=np.asarray(params[0]['A']);b=np.zeros(K);Q=np.eye(K)-A@A.T;C=np.eye(K);d=np.zeros(K);m1=np.zeros(K);P1=np.eye(K)
  native=jax.jit(lambda obs:trainer.apply(TrainData(obs=(obs,)))[1])(jnp.asarray(x))
  nf=np.asarray(native.params['means']);nc=np.asarray(native.params['covs'])
  filt,fc,sm,sc=kalman(pseudo,A,b,Q,C,d,R,m1,P1)
 else:
  A=np.asarray(params.dynamics.weights);b=np.asarray(params.dynamics.bias);Q=np.asarray(params.dynamics.cov)
  C=np.asarray(params.emissions.weights);d=np.asarray(params.emissions.bias);R=np.asarray(params.emissions.cov)
  m1=np.asarray(params.initial.mean);P1=np.asarray(params.initial.cov)
  model=LinearGaussianSSM(K,x.shape[-1])
  native=jax.jit(jax.vmap(lambda obs:model.smoother(params,obs)))(jnp.asarray(x))
  nf=np.asarray(native.smoothed_means);nc=np.asarray(native.smoothed_covariances)
  filt,fc,sm,sc=kalman(x,A,b,Q,C,d,R,m1,P1)
 nc_for_check=nc-reported_jitter*np.eye(K)
 discrepancy={'smoother_mean_max_abs':float(np.max(abs(nf-sm))),'smoother_cov_max_abs':float(np.max(abs(nc_for_check-sc))),'endpoint_mean_max_abs':float(np.max(abs(nf[:,-1]-filt[:,-1]))),'endpoint_cov_max_abs':float(np.max(abs(nc_for_check[:,-1]-fc[:,-1])))}
 assert max(discrepancy.values())<1e-7,discrepancy
 return {'smooth_mean':nf,'smooth_cov':nc,'filter_mean':filt,'filter_cov':fc,'A':A,'b':b,'Q':Q,'check':discrepancy}

def affine(x,y):
 xx=x.reshape(-1,x.shape[-1]);yy=y.reshape(-1,y.shape[-1]);design=np.c_[xx,np.ones(len(xx))]
 wb,_,rank,singular=np.linalg.lstsq(design,yy,rcond=None)
 return wb[:-1],wb[-1],{'design_rank':int(rank),'design_condition':float(singular[0]/singular[-1])}
def r2(y,p):
 yy=y.reshape(-1,y.shape[-1]);pp=p.reshape(-1,p.shape[-1]);den=((yy-yy.mean(0))**2).sum(0)
 return float(np.mean(1-((yy-pp)**2).sum(0)/den))
def propagate(m,P,A,b,Q,maxh):
 ms=[m.copy()];ps=[P.copy()]
 for _ in range(maxh):
  m=m@A.T+b;P=A@P@A.T+Q;ms.append(m.copy());ps.append(P.copy())
 return np.stack(ms,1),np.stack(ps,1)
def oracle(arrays,split):
 return kalman(arrays[split+'_x'],arrays['true_A'],arrays['true_b'],arrays['true_Q'],arrays['true_C'][0],arrays['true_d'][0],arrays['true_R'],arrays['true_m1'],arrays['true_Q1'])

def main():
 source_check();seal=OUT/'SELECTION_SEAL54_V1.json';sealed=json.loads(seal.read_text());assert len(sealed['fits'])==12
 for fit in sealed['fits']:assert sha(fit['checkpoint'])==fit['checkpoint_sha256']
 assert not (OUT/'evaluation54.json').exists(),'Keep completed evaluation immutable'
 start=time.perf_counter();prefix=PLAN['prefix_length'];hs=PLAN['horizons'];maxh=max(hs)
 readout_records=[];checks=[]
 # Phase 1: no heldout signal or states are read; freeze every diagnostic readout.
 for fit in sealed['fits']:
  run=fit['run'];method=fit['method'];task=next(t['name'] for t in PLAN['tasks'] if t['name'] in run);seed=int(run.rsplit('seed',1)[1])
  with np.load(data_path(task)) as ar:tx=ar['train_x'];tz=ar['train_z']
  params=pickle.loads(pathlib.Path(fit['checkpoint']).read_bytes());comp=components(method,task,seed,params,tx)
  Wz,bz,zinfo=affine(comp['smooth_mean'],tz);Wx,bx,xinfo=affine(comp['smooth_mean'],tx)
  readout=CKPT/(run+'_diagnostic_readout.npz');assert not readout.exists()
  np.savez_compressed(readout,Wz=Wz,bz=bz,Wx=Wx,bx=bx)
  rec={'run':run,'method':method,'task':task,'seed':seed,'file':str(readout),'sha256':sha(readout),'state_map':zinfo,'observation_map':xinfo,'train_state_R2':r2(tz,comp['smooth_mean']@Wz+bz)}
  readout_records.append(rec);checks.append({'run':run,'split':'train_full','errors':comp['check']})
 readout_seal=OUT/'READOUT_SEAL54_V1.json';assert not readout_seal.exists()
 dump(readout_seal,{'utc':utc(),'status':'ALL_TRAINING_ONLY_AFFINE_DIAGNOSTIC_READOUTS_FROZEN_BEFORE_HELDOUT_EVALUATION','fit_seal_sha256':sha(seal),'readouts':readout_records})
 # Phase 2: all original model and training-only diagnostic mappings are immutable.
 records=[];oracle_records=[]
 for task in PLAN['tasks']:
  name=task['name'];a=load_data(name);hx=a['heldout_x'];hz=a['heldout_z'];K=task['latent_dim'];N=len(hx)
  fm,fp,sm,sp=oracle(a,'heldout')
  # Use only the first 50 observations for the oracle's forecasting state.
  of,op,_,_=kalman(hx[:,:prefix],a['true_A'],a['true_b'],a['true_Q'],a['true_C'][0],a['true_d'][0],a['true_R'],a['true_m1'],a['true_Q1'])
  om,oc=propagate(of[:,-1],op[:,-1],a['true_A'],a['true_b'],a['true_Q'],maxh)
  om=om[:,hs];oc=oc[:,hs];truth=hz[:,[prefix-1+h for h in hs]];xtruth=hx[:,[prefix-1+h for h in hs]]
  ox=om@a['true_C'][0].T+a['true_d'][0]
  # Conditional Bayes errors and a prior-mean control provide scale, not learned competitors.
  oseq=((om-truth)**2).mean(-1);oxseq=((ox-xtruth)**2).mean(-1)
  np.savez_compressed(ART/(name+'_oracle.npz'),state_mean=om,state_cov=oc,observation_mean=ox,state_truth=truth,observation_truth=xtruth,state_sequence_mse=oseq,observation_sequence_mse=oxseq)
  oracle_records.append({'task':name,'full_smoothing_R2':r2(hz,sm),'horizons':[{'h':h,'state_mse':float(oseq[:,i].mean()),'observation_mse':float(oxseq[:,i].mean()),'zero_state_mse':float((truth[:,i]**2).mean())} for i,h in enumerate(hs)]})
  for rd in [r for r in readout_records if r['task']==name]:
   run=rd['run'];fit=next(f for f in sealed['fits'] if f['run']==run);params=pickle.loads(pathlib.Path(fit['checkpoint']).read_bytes())
   with np.load(rd['file']) as rr:Wz=rr['Wz'];bz=rr['bz'];Wx=rr['Wx'];bx=rr['bx']
   full=components(rd['method'],name,rd['seed'],params,hx)
   ctx=components(rd['method'],name,rd['seed'],params,hx[:,:prefix])
   checks.extend([{'run':run,'split':'heldout_full','errors':full['check']},{'run':run,'split':'heldout_causal_prefix','errors':ctx['check']}])
   lm,lc=propagate(ctx['smooth_mean'][:,-1],ctx['smooth_cov'][:,-1],ctx['A'],ctx['b'],ctx['Q'],maxh)
   # Verify recurrence against closed-form powers for each requested horizon.
   power_error=0.
   for h in hs:
    Ah=np.linalg.matrix_power(ctx['A'],h);bias=sum((np.linalg.matrix_power(ctx['A'],j)@ctx['b'] for j in range(h)),np.zeros(K))
    power_error=max(power_error,float(np.max(abs(lm[:,h]-(ctx['smooth_mean'][:,-1]@Ah.T+bias)))))
   assert power_error<1e-9
   if rd['method']=='rp':
    # Unconditional upstream Trainer.rollout assumes ACLG attributes; use the actual
    # native transition-chain component, with an explicit arbitrary initial belief.
    pp={'A':jnp.asarray(ctx['A']),'b':jnp.asarray(ctx['b']),'Q':jnp.asarray(ctx['Q']),'m1':jnp.asarray(ctx['smooth_mean'][0,-1]),'Q1':jnp.asarray(ctx['smooth_cov'][0,-1])}
    chain=transitions_to_marginals(pp,maxh+1,invariant=False)
    chain_error=max(float(np.max(abs(np.asarray(chain.params['means'])-lm[0]))),float(np.max(abs(np.asarray(chain.params['covs'])-lc[0]))))
    assert chain_error<1e-9
   else:chain_error=None
   pred=lm[:,hs]@Wz+bz;xp=lm[:,hs]@Wx+bx;zseq=((pred-truth)**2).mean(-1);xseq=((xp-xtruth)**2).mean(-1);oseqdiff=((pred-om)**2).mean(-1)
   Wval,bval,_=affine(full['smooth_mean'],hz)
   rank=int(np.linalg.matrix_rank(Wz));condition=float(np.linalg.cond(Wz));Aeig=np.linalg.eigvals(ctx['A']);Beig=np.linalg.eigvals(a['true_A']);ri,ci=linear_sum_assignment(abs(Aeig[:,None]-Beig[None,:]));eigerr=float(abs(Aeig[ri]-Beig[ci]).mean())
   Aphys=Wz.T@ctx['A']@np.linalg.inv(Wz.T) if rank==K else None
   artifact=ART/(run+'_heldout.npz');assert not artifact.exists()
   np.savez_compressed(artifact,state_mean=pred,auxiliary_observation_mean=xp,state_sequence_mse=zseq,observation_sequence_mse=xseq,conditional_mean_sequence_mse=oseqdiff,full_smoothing_state_mean=full['smooth_mean']@Wz+bz,latent_A=ctx['A'],physical_A=Aphys if Aphys is not None else np.zeros((0,0)))
   row={'run':run,'method':rd['method'],'task':name,'seed':rd['seed'],'artifact':str(artifact),'artifact_sha256':sha(artifact),'checkpoint_sha256':fit['checkpoint_sha256'],'readout_sha256':rd['sha256'],
    'full_smoothing_R2_train_readout':r2(hz,full['smooth_mean']@Wz+bz),'full_smoothing_R2_within_heldout_refit_diagnostic_only':r2(hz,full['smooth_mean']@Wval+bval),'state_map_rank':rank,'state_map_condition':condition,'transition_eigenvalue_mean_absolute_error':eigerr,'physical_A_relative_frobenius_error':float(np.linalg.norm(Aphys-a['true_A'])/np.linalg.norm(a['true_A'])) if Aphys is not None else None,'latent_eigenvalues':[[float(x.real),float(x.imag)] for x in Aeig],'true_eigenvalues':[[float(x.real),float(x.imag)] for x in Beig],'power_check_error':power_error,'native_transition_chain_check_error':chain_error,
    'horizons':[{'h':h,'state_mse':float(zseq[:,i].mean()),'state_R2':r2(truth[:,i],pred[:,i]),'auxiliary_observation_mse':float(xseq[:,i].mean()),'conditional_mean_discrepancy':float(oseqdiff[:,i].mean()),'per_dimension_state_mse':((pred[:,i]-truth[:,i])**2).mean(0).tolist()} for i,h in enumerate(hs)]}
   records.append(row);print(json.dumps({'run':run,'R2':row['full_smoothing_R2_train_readout'],'eigen_error':eigerr,'h10_state_mse':next(r['state_mse'] for r in row['horizons'] if r['h']==10)}),flush=True)
 # Check that evaluation did not alter any selected model or diagnostic mapping.
 for fit in sealed['fits']:assert sha(fit['checkpoint'])==fit['checkpoint_sha256']
 for rd in readout_records:assert sha(rd['file'])==rd['sha256']
 dump(OUT/'verification54.json',{'utc':utc(),'status':'PASS_INDEPENDENT_NUMPY_KALMAN_AND_ROLLOUT_ARITHMETIC','native_smoother_comparisons':checks,'comparison_count':len(checks),'max_absolute_smoother_discrepancy':max(max(r['errors'].values()) for r in checks),'all_12_checkpoint_hashes_unchanged':True,'all_12_training_only_readout_hashes_unchanged':True,'source_files_unchanged':source_check(),'evaluation_script_sha256':sha(__file__),'explicit_upstream_jitter':'RP dists_utils.parallel_smoother unconditionally adds 1e-5 I to reported smoothed covariance; comparison subtracts exactly that documented term. Actual native reported covariances remain used in rollout.','scope':'Independent numerical equations verify the native Gaussian surrogate posterior; they do not prove that the RPM surrogate equals the proper ratio-model posterior.'})
 dump(OUT/'evaluation54.json',{'utc':utc(),'fit_seal_sha256':sha(seal),'readout_seal_sha256':sha(readout_seal),'plan_sha256':sha(PLAN_PATH),'records':records,'oracle':oracle_records,'runtime_seconds':time.perf_counter()-start,'information':'All learned fits use train observations only. State-supervised affine mappings are diagnostic after training, never a deployable unsupervised decoder. Future observations are unavailable to both learned and oracle forecasting states.','scope':'Two in-model linear synthetic tasks; fixed data draw each, three training initializations. Not new method evidence.'})
 print('EVALUATION_COMPLETE',flush=True)
if __name__=='__main__':main()
