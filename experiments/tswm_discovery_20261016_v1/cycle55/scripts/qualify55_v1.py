"""Native finite-bank and independent quadrature checks before population profiling."""
from common55 import *
import numpy as np
import jax,jax.numpy as jnp,jax.random as jr
from scipy.special import roots_hermitenorm,logsumexp
from scipy.stats import norm
from population55_v1 import finite,population,posterior_terms,sigma_point_population
from rp_ssm.dists import LGStationaryParam,GaussianNatParam,LGChainDistParam
from rp_ssm.rpm import RPSSM,ConstrainedIVFreeEnergy
from rp_ssm.dists_utils import parallel_smoother

def native(a,lam,b,x):
    N,T=x.shape
    p=LGStationaryParam(start_from_invariant=True,stay_at_invariant=True,opt_params=['A'],A=jnp.array([[a]]))
    params=p.init(jr.PRNGKey(55),None);p=p.update(params);chain,inference=p.to_chain(T,None)
    assert np.max(abs(np.asarray(params['A'])-a))<1e-15
    sites=GaussianNatParam(p=jnp.full((1,N,T,1,1),lam),pwm=jnp.asarray((lam*b*x)[None,...,None]))
    one=GaussianNatParam(p=sites.params['p'][0],pwm=sites.params['pwm'][0]).dist_param
    means,covs,cross=jax.vmap(lambda f:parallel_smoother(inference,f,1))(one)
    posterior=LGChainDistParam(means=means,covs=covs,cross_covs=cross)
    terms=ConstrainedIVFreeEnergy(RPSSM(p,[])).get_loss_terms(chain,sites,posterior)
    kf,gamma,kp=[float(v) for v in terms]
    return {'loss':-(gamma-kf-kp)/(N*T),'means':np.asarray(means)[...,0],'cov':np.asarray(covs)[...,0,0],'cross':np.asarray(cross)[...,0,0]}

def main():
    pins=source_check();rng=np.random.default_rng(5501);x=rng.normal(size=(7,6));native_checks=[]
    for a,lam,b in [(.2,.8,.7),(.7,2.,1.1),(-.4,3.,.4),(.95,5.,1.)]:
        n=native(a,lam,b,x);f=finite(a,lam,b,x)
        errors={'loss':abs(n['loss']-f['loss']),'mean':float(np.max(abs(n['means']-f['means']))),'cov':float(np.max(abs(n['cov']-f['covariances'][None]))),'adjacent_cov':float(np.max(abs(n['cross']-f['cross_covariances'][None])))}
        assert max(errors.values())<1e-7,errors
        native_checks.append({'parameters':[a,lam,b],'errors':errors,'source_loss':n['loss'],'independent_loss':f['loss']})
    a,lam,b,a_star,r,T=.35,2.,.7,.6,.3,6
    _,_,C,_,_,_=posterior_terms(a,lam,b,T,0.)
    nodes,weights=roots_hermitenorm(128);logw=np.log(weights)-.5*np.log(2*np.pi);u=b*np.sqrt(1+r)*nodes;integrals=[]
    for t in [0,2,5]:
        for m in [-.4,.2,.7]:
            logs=norm.logpdf(u,loc=m,scale=np.sqrt(C[t]+1/lam))-norm.logpdf(u,loc=0,scale=np.sqrt(1+1/lam))
            numeric=float(logsumexp(logw+logs));coef=lam/(1+lam*C[t]);quad=.5*lam*lam*(C[t]-1)/((1+lam*C[t])*(1+lam));tau=b*b*(1+r);D=1-2*quad*tau
            constant=.5*np.log((1+lam)/(1+lam*C[t]))-.5*coef*m*m
            exact=constant-.5*np.log(D)+.5*coef*coef*m*m*tau/D
            error=abs(numeric-exact);assert error<1e-9
            integrals.append(error)
    target_errors=[]
    for jitter in [0.,1e-5]:
        exact=population(a,lam,b,a_star,r,T,jitter)['loss'];cubature=sigma_point_population(a,lam,b,a_star,r,T,jitter)
        error=abs(exact-cubature);assert error<1e-9;target_errors.append(error)
    equalities=[]
    for aa in [.5,.95]:
        for rr in [.03,.3,3.]:
            for tt in [6,24]:
                value=population(aa,1/rr,1.,aa,rr,tt,0.)
                error=max(abs(value['reference_variance']-1),abs(value['bound_gap']),abs(value['loss']-value['true_likelihood_optimum']))
                assert error<1e-8,(aa,rr,tt,value)
                equalities.append({'a_star':aa,'r':rr,'T':tt,'error':error,'loss':value['loss']})
    assert source_check()==pins
    record={'utc':utc(),'status':'PASS_NATIVE_FINITE_BANK_AND_INDEPENDENT_POPULATION_QUADRATURE','native_finite_checks':native_checks,'reference_quadrature_max_error':max(integrals),'target_sigma_point_max_error':max(target_errors),'true_point_equalities':equalities,'formal_C54_model_or_data_reads':0,'optimizer_updates':0,'native_source_sha256':pins,'protocol_sha256':sha(ROOT/'PROTOCOL.md'),'math_script_sha256':sha(ROOT/'scripts/population55_v1.py'),'qualification_script_sha256':sha(__file__),'scope':'Scalar Gaussian source-objective audit; no native training, nonlinear benchmark, theorem refutation or new-method evidence.'}
    dump(OUT/'QUALIFICATION55_V1.json',record)
    print(json.dumps({'status':record['status'],'native_loss_max_error':max(x['errors']['loss'] for x in native_checks),'quadrature_max_error':max(integrals),'target_expectation_error':max(target_errors),'true_point_max_error':max(x['error'] for x in equalities)}),flush=True)

if __name__=='__main__':main()
