"""Fixed normalized RPM: ordinary Gaussian VI versus Gaussian interior bound.

No learned states, unknown dynamics, reconstruction error or stochastic optimizer
is involved. The counterexample separates the Gaussian family from the extra
auxiliary-bound constraint. It does NOT by itself prove a trained RP-GSSM fails.
"""
import hashlib,json,pathlib,time
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp,roots_hermitenorm
from scipy.stats import norm
ROOT=pathlib.Path(__file__).resolve().parents[1]

def log_normal(z,v):return -.5*(np.log(2*np.pi*v)+z*z/v)
def log_unnormalized(z,V,eps):
    z=np.asarray(z);p=log_normal(z,1.);f=log_normal(z,V)
    F=np.logaddexp(np.log1p(-eps)+p,np.log(eps)+f)
    return p+f-F

def solve(V,eps,orders):
    density=lambda z:np.exp(log_unnormalized(z,V,eps))
    Z,zerr=quad(density,-np.inf,np.inf,epsabs=1e-11,epsrel=1e-11,limit=300)
    second,verr=quad(lambda z:z*z*density(z),-np.inf,np.inf,epsabs=1e-10,epsrel=1e-10,limit=300)
    variance=second/Z;logZ=np.log(Z)
    ordinary=[]
    for order in orders:
        nodes,weights=roots_hermitenorm(order);weights=weights/np.sqrt(2*np.pi)
        def elbo(logv):
            v=np.exp(logv)
            return float(weights@log_unnormalized(nodes*np.sqrt(v),V,eps)+.5*np.log(2*np.pi*np.e*v))
        res=minimize_scalar(lambda lv:-elbo(lv),bounds=(-8,np.log(4*V)),method='bounded',options={'xatol':1e-11})
        ordinary.append({'quadrature_order':order,'variance':float(np.exp(res.x)),'kl_to_exact':float(logZ+res.fun)})
    # Unnormalized auxiliary g(z)=exp(-h*z²/2). Integrability requires h>-1/V.
    # For fixed g, exact q optimum is sqrt(prior*f_rare*g): v=2/(1+1/V+h).
    # Both centered component variances imply the optimal auxiliary linear term
    # is zero; hence this one-dimensional optimization covers the symmetric optimum.
    def interior(t):
        h=-1/V+np.exp(t);P=1+1/V+h
        logGamma=logsumexp([np.log1p(-eps)-.5*np.log1p(h),np.log(eps)-.5*np.log1p(h*V)])
        G=np.log(2)-.5*np.log(V)-np.log(P)-logGamma
        return G,h,2/P
    result=minimize_scalar(lambda t:-interior(t)[0],bounds=(-27,10),method='bounded',options={'xatol':1e-11})
    G,h,v=interior(result.x)
    caps={}
    for label,var in [('positive_site_capacity_bound',1.),('interior_capacity_bound',2.),('interior_optimized',v),('gaussian_vi',ordinary[-1]['variance'])]:
        interval=1.959963984540054*np.sqrt(var)
        coverage=quad(density,-interval,interval,epsabs=1e-10)[0]/Z
        caps[label]={'variance':float(var),'true_coverage_of_gaussian_95_interval':float(coverage),
                     'predicted_abs_state_gt2_probability':float(2*norm.sf(2/np.sqrt(var)))}
    true_tail=1-quad(density,-2,2,epsabs=1e-10)[0]/Z
    # The implied proper categorical observation likelihood is eps*f_rare/F.
    # Its marginal need not equal the empirical reference eps at finite mismatch.
    # It is eps*Z, which is reported rather than silently assumed identical.
    return {'recognition_variance':V,'reference_rare_probability':eps,
            'implied_observation_rare_probability':float(eps*Z),'exact_variance':float(variance),
            'exact_tail_probability_abs_state_gt2':float(true_tail),'exact_log_evidence_without_log_reference':float(logZ),
            'ordinary_gaussian_vi':ordinary,'interior_auxiliary_precision':float(h),'interior_q_variance':float(v),
            'interior_objective_gap':float(logZ-G),'capacity_controls':caps,
            'integration_error_Z':float(zerr),'integration_error_second_moment':float(verr),
            'reference_mixture_tv_upper_bound':eps,
            'unimodality':'strictly log-concave in this centered V>=1 construction; not a multimodality example'}

def main():
    pp=ROOT/'outputs/rpm_tail_plan.json';plan=json.loads(pp.read_text());start=time.perf_counter();records=[]
    for V in plan['rare_recognition_variances']:
      for eps in plan['rare_reference_probabilities']:
        row=solve(V,eps,plan['gauss_hermite_orders']);records.append(row)
        assert row['interior_q_variance']<2+1e-10
        assert row['ordinary_gaussian_vi'][-1]['kl_to_exact']>=-1e-8
        assert row['interior_objective_gap']>=row['ordinary_gaussian_vi'][-1]['kl_to_exact']-1e-7
        print(json.dumps({k:row[k] for k in ['recognition_variance','reference_rare_probability','exact_variance','interior_q_variance','interior_objective_gap']}),flush=True)
    result={'plan_sha256':hashlib.sha256(pp.read_bytes()).hexdigest(),'code_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'records':records,'runtime_seconds':time.perf_counter()-start,
            'scope':'new fixed-model mathematical audit using exact quadrature and ordinary Gaussian VI, not a trained-RP reproduction'}
    (ROOT/'outputs/rpm_tail_probe.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
if __name__=='__main__':main()
