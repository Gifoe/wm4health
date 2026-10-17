"""One API/arithmetic qualification on deterministic dummy observations, outside evidence."""
from common54 import *
from evaluate54_v2 import components,propagate
from rp_ssm.datasets import TrainData
from dynamax.linear_gaussian_ssm import LinearGaussianSSM
def main():
 source_check();K=3;D=5
 x=np.sin(np.arange(2*12*D).reshape(2,12,D)/17.)+.1*np.random.default_rng(54).normal(size=(2,12,D))
 energy,cfg=rp_model('linear_small',2021)
 rp,_,_=energy.init(jr.PRNGKey(54),TrainData(obs=(jnp.asarray(x),)),cfg)
 rpcomp=components('rp','linear_small',2021,rp,x)
 model=LinearGaussianSSM(K,D);ep,props=model.initialize(jr.PRNGKey(54),dynamics_weights=.8*jnp.eye(K),dynamics_covariance=.2*jnp.eye(K),emission_covariance=.3*jnp.eye(D))
 ep,history=model.fit_em(ep,props,jnp.asarray(x),num_iters=2,verbose=False)
 ec=components('em','linear_small',2021,ep,x)
 assert np.isfinite(np.asarray(history)).all()
 m,c=propagate(rpcomp['smooth_mean'][:,-1],rpcomp['smooth_cov'][:,-1],rpcomp['A'],rpcomp['b'],rpcomp['Q'],10)
 assert m.shape==(2,11,3) and c.shape==(2,11,3,3)
 dump(OUT/'QUALIFICATION54_V2.json',{'utc':utc(),'status':'PASS_DUMMY_NATIVE_API_AND_INDEPENDENT_KALMAN_INTEGRATION','data':'Deterministic 2x12x5 sine array with fixed-seed full-rank Gaussian perturbation, no formal dataset read','prior_qualification_failure':'V1 compared native jittered RP covariance to unjittered equations: mean discrepancy 1.20e-10, covariance discrepancy 1.00003054385e-5. Source explicitly adds 1e-5 I; V2 checks that exact documented difference. No formal model or training objective changed.','formal_training_updates':0,'dummy_EM_iterations':2,'rp_discrepancy':rpcomp['check'],'em_discrepancy':ec['check'],'script_sha256':sha(__file__),'evaluation_script_sha256':sha(pathlib.Path(__file__).with_name('evaluate54_v2.py')),'scope':'Critical changed executable paths only. Not an experiment result or evidence of model quality.'})
 print('QUALIFICATION_PASS',flush=True)
if __name__=='__main__':main()
