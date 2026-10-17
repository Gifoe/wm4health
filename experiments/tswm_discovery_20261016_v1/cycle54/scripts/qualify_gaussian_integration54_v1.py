"""Native initial-belief -> rollout -> affine Gaussian score on dummy inputs only."""
from common54 import *
import ctypes
from scipy.stats import multivariate_normal, chi2
from evaluate54_v2 import components, propagate
from gaussian_belief_math54_v1 import scores
from rp_ssm.datasets import TrainData
from dynamax.linear_gaussian_ssm import LinearGaussianSSM

def main():
    source_check()
    fn=ctypes.windll.kernel32.SetProcessAffinityMask;fn.argtypes=[ctypes.c_void_p,ctypes.c_size_t];assert fn(ctypes.c_void_p(-1),ctypes.c_size_t(0xf))
    records=[];rng=np.random.default_rng(54882);N=2;T=12
    for task in PLAN['tasks']:
        name=task['name'];K=task['latent_dim'];D=task['emission_dim']
        x=np.sin(np.arange(N*T*D).reshape(N,T,D)/17)+.1*rng.normal(size=(N,T,D))
        energy,cfg=rp_model(name,2021)
        rp,_,_=energy.init(jr.PRNGKey(54882),TrainData(obs=(jnp.asarray(x),)),cfg)
        em_model=LinearGaussianSSM(K,D)
        ep,_=em_model.initialize(jr.PRNGKey(54882),dynamics_weights=.8*jnp.eye(K),dynamics_covariance=.2*jnp.eye(K),emission_covariance=.3*jnp.eye(D))
        for method,params in [('rp',rp),('em',ep)]:
            c=components(method,name,2021,params,x)
            lm,lc=propagate(c['smooth_mean'][:,-1],c['smooth_cov'][:,-1],c['A'],c['b'],c['Q'],max(PLAN['horizons']))
            W=np.eye(K)+.05*rng.normal(size=(K,K));bias=rng.normal(size=K)
            mm=lm[:,PLAN['horizons']]@W+bias;cm=W.T@lc[:,PLAN['horizons']]@W
            mo=np.zeros_like(mm);co=np.broadcast_to(.7*np.eye(K),cm.shape).copy();truth=rng.normal(size=mm.shape)
            value=scores(mm,cm,mo,co,truth,float(chi2.ppf(.95,K)))
            assert value['gaussian_kl'].shape==(N,6) and np.all(value['valid_covariance'])
            direct=np.empty((N,6))
            for n in range(N):
                for j in range(6):direct[n,j]=-multivariate_normal.logpdf(truth[n,j],mean=mm[n,j],cov=cm[n,j])
            error=float(np.max(abs(value['learned_nll']-direct)));assert error<1e-9
            records.append({'task':name,'method':method,'score_shape':[N,6],'physical_covariance_shape':list(cm.shape),'scipy_nll_max_error':error,'native_inference_checks':c['check']})
    source_check()
    record={'utc':utc(),'status':'PASS_DUMMY_NATIVE_INITIAL_INFERENCE_ROLLOUT_AFFINE_AND_BATCHED_GAUSSIAN_SCORE','records':records,'formal_signal_state_or_checkpoint_reads':0,'optimizer_updates':0,'EM_iterations':0,'native_source_unchanged':True,'script_sha256':sha(__file__),'math_script_sha256':sha(ROOT/'scripts/gaussian_belief_math54_v1.py'),'existing_component_script_sha256':sha(ROOT/'scripts/evaluate54_v2.py'),'scope':'New independently initialized dummy models only; no formal fit, selected model or performance evidence.'}
    p=OUT/'QUALIFICATION_GAUSSIAN_INTEGRATION54_V1.json';assert not p.exists();dump(p,record)
    print(json.dumps(record),flush=True)

if __name__=='__main__':main()
