"""Frozen-score estimator controls: known fixes, no new training/heldout evidence."""
from common54 import *
import ctypes,pickle,time
from scipy.special import logsumexp
from audit_training_objective54_v2 import objective
from per_target_auxiliary54_v3 import per_target_terms
fn=ctypes.windll.kernel32.SetProcessAffinityMask;fn.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
assert fn(ctypes.c_void_p(-1),ctypes.c_size_t(0xff000000))
def estimate(base,G,dG,target,reference,mode,logu=None):
 score=G[:,target[:,None],reference[None,:]];ds=dG[:,target[:,None],reference[None,:]]
 if mode=='loo':
  score=score.copy();score[:,np.arange(32),np.arange(32)]=-np.inf
 if mode in ['native','loo','independent']:
  weights=np.exp(score-logsumexp(score,axis=-1,keepdims=True));denom=(weights*ds).sum(-1)
 else:
  weights=np.exp(score-logu[:,target,None]);denom=(weights*ds).mean(-1)
 return float(base[target].mean()+denom.mean())
def main():
 source_check();out=OUT/'normalizer_control_audit_v1';assert not out.exists();out.mkdir();records=[];start=time.perf_counter()
 with np.load(data_path('linear_small')) as ar:x=ar['train_x']
 for seed in [0,1]:
  run=f'rp_linear_small_seed{seed}';r=json.loads((OUT/'runs'/run/'result.json').read_text());path=pathlib.Path(r['checkpoint']);before=sha(path);assert before==r['checkpoint_sha256'];params=pickle.loads(path.read_bytes());A=jnp.asarray(params[0]['A']);xx=jnp.asarray(x)
  energy,_=objective('linear_small',seed,200)
  fn=jax.jit(lambda alpha:per_target_terms(energy,({'A':alpha*A},*params[1:]),xx))
  (loss,G),(d_loss,dG)=jax.jvp(fn,(jnp.asarray(1.),),(jnp.asarray(1.),))
  G=np.asarray(G);dG=np.asarray(dG);loss=np.asarray(loss);d_loss=np.asarray(d_loss);assert G.shape==(100,200,200)
  assert all(np.isfinite(v).all() for v in [G,dG,loss,d_loss]);lu=logsumexp(G,axis=-1)-np.log(200)
  full_den=(np.exp(G-logsumexp(G,axis=-1,keepdims=True))*dG).sum(-1).mean(0);base=d_loss-full_den
  previous=json.loads((OUT/'training_objective_audit_v2'/f'{run}.json').read_text());full_derivative=float(d_loss.mean());jvp_error=abs(full_derivative-previous['full_alpha_derivative']);assert jvp_error<1e-7,jvp_error
  with np.load(ART/'reference_control_audit_v2'/f'{run}.npz') as ar:
   balanced=ar['balanced_indices'];native_grads=ar['batch_reference_gradients']
  reconstructed=np.array([estimate(base,G,dG,ids,ids,'native') for ids in balanced]);native_directions=np.einsum('nkl,kl->n',native_grads,np.asarray(A));cache_error=float(np.max(abs(reconstructed-native_directions)));assert cache_error<1e-6,cache_error
  rng=np.random.default_rng(5426);logstates={.9:np.full((100,200),-np.inf),.1:np.full((100,200),-np.inf)}
  names=['native_coupled','leave_designated_diagonal_out','independent_reference','oracle_frozen_normalizer','running_normalizer_gamma09','running_normalizer_gamma01'];values=[];target_rows=[];reference_rows=[]
  for it in range(512+1024):
   target=rng.integers(0,200,size=32);ref=rng.integers(0,200,size=32)
   independent_scores=G[:,target[:,None],ref[None,:]];sample_logmean=logsumexp(independent_scores,axis=-1)-np.log(32)
   unique,first=np.unique(target,return_index=True)
   for gamma,state in logstates.items():
    prev=state[:,unique];cur=sample_logmean[:,first]
    state[:,unique]=np.where(np.isneginf(prev),cur,np.logaddexp(np.log1p(-gamma)+prev,np.log(gamma)+cur))
   row=[estimate(base,G,dG,target,target,'native'),estimate(base,G,dG,target,target,'loo'),estimate(base,G,dG,target,ref,'independent'),estimate(base,G,dG,target,ref,'oracle',lu),estimate(base,G,dG,target,ref,'running',logstates[.9]),estimate(base,G,dG,target,ref,'running',logstates[.1])]
   assert np.isfinite(row).all()
   if it>=512:values.append(row);target_rows.append(target);reference_rows.append(ref)
  values=np.asarray(values);artifact=ART/'normalizer_control_audit_v1';artifact.mkdir(exist_ok=True)
  cache=artifact/(run+'_scores.npz');ap=artifact/(run+'_estimates.npz');assert not cache.exists() and not ap.exists()
  np.savez_compressed(cache,score=G,directional_score_derivative=dG,loss_per_target=loss,directional_loss_per_target=d_loss,base_directional_derivative=base)
  np.savez_compressed(ap,estimator_values=values,target_indices=np.stack(target_rows),reference_indices=np.stack(reference_rows),balanced_native_reconstructed=reconstructed,balanced_native_direct=native_directions)
  summaries=[]
  for i,name in enumerate(names):
   v=values[:,i];mean=float(v.mean());std=float(v.std(ddof=1));summaries.append({'estimator':name,'alpha_derivative_mean':mean,'conditional_bias_vs_full_bank':mean-full_derivative,'std_over_batches':std,'iid_MC_SE':std/np.sqrt(len(v)) if i<4 else None,'MC_scope':'Conditional on fixed training data and frozen model. Running estimators have temporal dependence; no iid MC interval claimed.'})
  record={'run':run,'checkpoint_sha256':before,'protocol_sha256':sha(ROOT/'TRAIN_ONLY_NORMALIZER_CONTROL54_V1.md'),'script_sha256':sha(__file__),'per_target_source_sha256':sha(pathlib.Path(__file__).with_name('per_target_auxiliary54_v3.py')),'full_bank_alpha_derivative':full_derivative,'jvp_vs_prior_native_autodiff_abs_error':jvp_error,'cached_vs_native_25_batch_max_abs_error':cache_error,'burn_in_batches':512,'recorded_batches':1024,'estimates':summaries,'score_cache':str(cache),'score_cache_sha256':sha(cache),'raw_estimates':str(ap),'raw_estimates_sha256':sha(ap),'checkpoint_unchanged':sha(path)==before,'heldout_read':False,'latent_truth_read':False,'optimizer_updates':0,'scope':'Estimator behavior at fixed parameters only, not official SogCLR reproduction, new method validation, parameter tracking convergence, or model forecasting accuracy.'}
  assert record['checkpoint_unchanged'];records.append(record);dump(out/(run+'.json'),record);print(json.dumps(record),flush=True)
 dump(out/'summary.json',{'utc':utc(),'status':'COMPLETE_FROZEN_NORMALIZER_ATTRIBUTION_AND_EXISTING_CONTROLS','records':records,'runtime_seconds':time.perf_counter()-start,'source_files_unchanged':source_check(),'formal_comparison_unchanged':True})
if __name__=='__main__':main()
