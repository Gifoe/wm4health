"""Reproduce first-update failure and independently verify the ordinary EM correction."""
from common54 import *
from corrected_em54_v1 import CorrectedLinearGaussianSSM
from dynamax.linear_gaussian_ssm import LinearGaussianSSM
from evaluate54_v2 import kalman
import inspect
def main():
 source_check()
 with np.load(data_path('linear_small')) as ar:x=ar['train_x']
 K,D=3,5;center=x.reshape(-1,D).mean(0);cov=np.cov(x.reshape(-1,D).T);e,U=np.linalg.eigh(cov);idx=np.argsort(e)[::-1];e=e[idx];U=U[:,idx]
 noise=max(float(e[K:].mean()),1e-3);rng=np.random.default_rng(5400)
 C=U[:,:K]*np.sqrt(np.maximum(e[:K]-noise,1e-3))[None,:];C=C+.05*np.sqrt(np.diag(cov))[:,None]*rng.normal(size=(D,K))/np.sqrt(K)
 A=.8*np.eye(K)+.02*rng.normal(size=(K,K))/np.sqrt(K);A=A*min(1.,.95/np.linalg.svd(A,compute_uv=False)[0])
 model=LinearGaussianSSM(K,D);fixed=CorrectedLinearGaussianSSM(K,D)
 p,props=model.initialize(jr.PRNGKey(0),initial_mean=jnp.zeros(K),initial_covariance=jnp.eye(K),dynamics_weights=jnp.asarray(A),dynamics_bias=jnp.zeros(K),dynamics_covariance=.2*jnp.eye(K),emission_weights=jnp.asarray(C),emission_bias=jnp.asarray(center),emission_covariance=noise*jnp.eye(D))
 stats,ll=jax.jit(jax.vmap(lambda obs:model.e_step(p,obs)))(jnp.asarray(x))
 old,_=model.m_step(p,props,stats,None);new,_=fixed.m_step(p,props,stats,None)
 _,_,sm,sc=kalman(x,A,np.zeros(K),.2*np.eye(K),C,center,noise*np.eye(D),np.zeros(K),np.eye(K))
 means=sm[:,0];independent=(sc[:,0]+means[:,:,None]*means[:,None,:]).mean(0)-np.outer(means.mean(0),means.mean(0))
 error=float(np.max(abs(np.asarray(new.initial.cov)-independent)));assert error<1e-7,error
 min_old=float(np.linalg.eigvalsh(np.asarray(old.initial.cov)).min());min_new=float(np.linalg.eigvalsh(np.asarray(new.initial.cov)).min());assert min_old<0 and min_new>0
 for v,w in zip(jax.tree.leaves(old.dynamics),jax.tree.leaves(new.dynamics)):assert np.array_equal(v,w)
 for v,w in zip(jax.tree.leaves(old.emissions),jax.tree.leaves(new.emissions)):assert np.array_equal(v,w)
 assert np.array_equal(old.initial.mean,new.initial.mean)
 _,history=fixed.fit_em(p,props,jnp.asarray(x),num_iters=3,verbose=False);history=np.asarray(history)
 assert np.isfinite(history).all() and np.min(np.diff(history))>0
 source=inspect.getsource(LinearGaussianSSM.m_step)
 dump(OUT/'QUALIFICATION_EM_FIX54_V1.json',{'utc':utc(),'status':'PASS_ORDINARY_EM_INITIAL_COVARIANCE_NORMALIZATION_FIX','training_data_only':True,'heldout_or_true_state_read':False,'original_first_mstep_initial_covariance_min_eigenvalue':min_old,'corrected_min_eigenvalue':min_new,'independent_numpy_expected_moment_max_abs_error':error,'all_other_mstep_parameters_identical':True,'corrected_three_iteration_training_loglik':history.tolist(),'native_mstep_source_sha256':hashlib.sha256(source.encode()).hexdigest(),'native_mstep_source_file':inspect.getsourcefile(LinearGaussianSSM),'native_mstep_source':source,'correction_source_sha256':sha(pathlib.Path(__file__).with_name('corrected_em54_v1.py')),'qualification_script_sha256':sha(__file__),'scope':'Routine implementation repair of the standard covariance moment, not a new method. No upstream source or global package modified. Four discarded training-only M-step executions (one direct, three in fit_em); the direct native and corrected calls compare the same E-step statistics.'})
 print('EM_INITIAL_COVARIANCE_FIX_QUALIFIED',min_old,min_new,error,history,flush=True)
if __name__=='__main__':main()
