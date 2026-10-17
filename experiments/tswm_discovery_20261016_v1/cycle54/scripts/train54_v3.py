"""Complete fixed-budget native RP and native LGSSM EM fits. No state access."""
from common54 import *
import argparse,pickle,time
from rp_ssm.datasets import TrainData
from rp_ssm.training import Trainer
from corrected_em54_v1 import CorrectedLinearGaussianSSM as LinearGaussianSSM

def main():
 p=argparse.ArgumentParser();p.add_argument('method',choices=['rp','em']);p.add_argument('task');p.add_argument('seed',type=int);a=p.parse_args()
 task=case(a.task)
 if a.method=='em':assert json.loads((OUT/'QUALIFICATION_EM_FIX54_V1.json').read_text())['status']=='PASS_ORDINARY_EM_INITIAL_COVARIANCE_NORMALIZATION_FIX'
 assert a.seed in PLAN['training_seeds'];source_check()
 run=f'{a.method}_{a.task}_seed{a.seed}';out=OUT/'runs'/run;out.mkdir(parents=True,exist_ok=True)
 assert not (out/'result.json').exists(),'Completed run must not be overwritten'
 datafile=data_path(a.task)
 with np.load(datafile) as archive:x=archive['train_x']
 # This training module never reads latent states or heldout arrays.
 x=jnp.asarray(x);start=time.perf_counter();log=out/'curve.jsonl'
 assert not log.exists(),'Preserve failed-attempt logs; use a versioned run'
 def emit(row):
  row['elapsed_seconds']=time.perf_counter()-start
  with log.open('a',encoding='utf-8') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
  print(json.dumps(row),flush=True)
 if a.method=='rp':
  energy,cfg=rp_model(a.task,a.seed);td=TrainData(obs=(x,))
  _,init_key=jr.split(jr.PRNGKey(a.seed));initial,_,_=energy.init(init_key,td,cfg)
  init_arrays=jax.tree.map(lambda v:np.asarray(v),initial)
  def logger(trainer,aux,indices):
   iteration=int(trainer.itr)+1
   if iteration==1 or iteration%100==0:
    value=float(trainer.loss_tot[-1]);assert np.isfinite(value)
    A=np.asarray(trainer.params[0]['A']);assert np.isfinite(A).all()
    emit({'iteration':iteration,'native_negative_auxiliary_free_energy':value,'A_singular_values':np.linalg.svd(A,compute_uv=False).tolist()})
   return {}
  trainer=Trainer(energy,cfg,logger);trainer.fit(td,use_pbar=False)
  params=jax.tree.map(lambda v:np.asarray(v),trainer.params)
  iterations=cfg.num_iter;history=np.asarray(trainer.loss_tot);assert np.isfinite(history).all()
  np.save(ART/(run+'_objective.npy'),history)
  configuration=cfg.__dict__.copy()
  configuration={k:str(v) if callable(v) else v for k,v in configuration.items()}
 else:
  K,D=task['latent_dim'],task['emission_dim'];xx=np.asarray(x);center=xx.reshape(-1,D).mean(0)
  cov=np.cov(xx.reshape(-1,D).T);e,U=np.linalg.eigh(cov);order=np.argsort(e)[::-1];e=e[order];U=U[:,order]
  noise=max(float(e[K:].mean()),1e-3);rng=np.random.default_rng(5400+a.seed)
  C=U[:,:K]*np.sqrt(np.maximum(e[:K]-noise,1e-3))[None,:]
  C=C+.05*np.sqrt(np.diag(cov))[:,None]*rng.normal(size=(D,K))/np.sqrt(K)
  A=.8*np.eye(K)+.02*rng.normal(size=(K,K))/np.sqrt(K)
  A=A*min(1.,.95/np.linalg.svd(A,compute_uv=False)[0])
  model=LinearGaussianSSM(K,D)
  params,props=model.initialize(jr.PRNGKey(a.seed),initial_mean=jnp.zeros(K),initial_covariance=jnp.eye(K),dynamics_weights=jnp.asarray(A),dynamics_bias=jnp.zeros(K),dynamics_covariance=.2*jnp.eye(K),emission_weights=jnp.asarray(C),emission_bias=jnp.asarray(center),emission_covariance=noise*jnp.eye(D))
  init_arrays=jax.tree.map(lambda v:np.asarray(v) if hasattr(v,'shape') else v,params)
  params,history=model.fit_em(params,props,x,num_iters=PLAN['em_iterations'],verbose=False)
  history=np.asarray(history)
  np.save(ART/(run+'_em_objective_audit_v3.npy'),history)
  (CKPT/(run+'_em_candidate_v3.pkl')).write_bytes(pickle.dumps(jax.tree.map(lambda v:np.asarray(v) if hasattr(v,'shape') else v,params)))
  dump(out/'EM_OBJECTIVE_AUDIT_V3.json',{'utc':utc(),'history':[float(v) if np.isfinite(v) else None for v in history],'finite':bool(np.isfinite(history).all()),'min_delta':float(np.min(np.diff(history))) if np.isfinite(history).all() else None,'elapsed_seconds':time.perf_counter()-start,'baseline_amendment_sha256':sha(ROOT/'plan54_em_amendment_v2.json')})
  assert np.isfinite(history).all();assert min(np.diff(history))>=-1e-5*max(1.,np.max(abs(history)))
  params=jax.tree.map(lambda v:np.asarray(v) if hasattr(v,'shape') else v,params)
  iterations=PLAN['em_iterations'];np.save(ART/(run+'_objective.npy'),history)
  for i in [0,9,49,99,199]:emit({'iteration':i+1,'native_training_log_marginal_likelihood':float(history[i])})
  configuration={'num_iters':iterations,'initialization':PLAN['em_initialization'],'noise_initialization':noise,'native_model':'Dynamax 0.1.5 LinearGaussianSSM with standard initial-covariance normalization fix','correction_source_sha256':sha(pathlib.Path(__file__).with_name('corrected_em54_v1.py')),'all_parameters_learned':True}
 assert all(np.isfinite(v).all() for v in jax.tree.leaves(params) if hasattr(v,'shape'))
 path=CKPT/(run+'.pkl');assert not path.exists();path.write_bytes(pickle.dumps(params))
 initpath=CKPT/(run+'_initial.pkl');assert not initpath.exists();initpath.write_bytes(pickle.dumps(init_arrays))
 dump(out/'result.json',{'utc':utc(),'status':'COMPLETE_FIXED_BUDGET','method':a.method,'task':a.task,'seed':a.seed,'iterations':iterations,'checkpoint':str(path),'checkpoint_sha256':sha(path),'initial_checkpoint_sha256':sha(initpath),'dataset_sha256':sha(datafile),'plan_sha256':sha(PLAN_PATH),'source_revision':PLAN['source_revision'],'baseline_amendment_sha256':sha(ROOT/'plan54_em_amendment_v2.json') if a.method=='em' else None,'script_sha256':sha(__file__),'common_sha256':sha(pathlib.Path(__file__).with_name('common54.py')),'runtime_seconds':time.perf_counter()-start,'configuration':configuration,'latent_states_read_in_training':False,'heldout_read_in_training':False,'checkpoint_selection':'terminal fixed-budget model','parameter_count':int(sum(np.asarray(v).size for v in jax.tree.leaves(params) if hasattr(v,'shape')))})
 print('COMPLETE',run,flush=True)
if __name__=='__main__':main()
