"""Stop before Monte Carlo if loss, derivatives or expectations disagree."""
from common56 import *
import numpy as np
import jax,jax.numpy as j,jax.random as jr
from math56 import sample_loss,population_loss,corr
from rp_ssm.dists import LGStationaryParam,GaussianNatParam,LGChainDistParam
from rp_ssm.rpm import RPSSM,ConstrainedIVFreeEnergy
from rp_ssm.dists_utils import parallel_smoother
sys.path.insert(0,str(PROJECT/'cycle55/scripts'))
from population55_v2 import population

def native(theta,x,fixed_q):
    N,T=x.shape;a=theta[0];lam=j.exp(theta[1]);b=j.exp(theta[2])
    p=LGStationaryParam(start_from_invariant=True,stay_at_invariant=True,opt_params=['A'],A=j.reshape(a,(1,1)))
    params=p.init(jr.PRNGKey(56),None);p=p.update(params);chain,inference=p.to_chain(T,None)
    sites=GaussianNatParam(p=j.full((1,N,T,1,1),lam),pwm=(lam*b*x)[None,...,None])
    one=GaussianNatParam(p=sites.params['p'][0],pwm=sites.params['pwm'][0]).dist_param
    means,covs,cross=jax.vmap(lambda f:parallel_smoother(inference,f,1))(one)
    posterior=LGChainDistParam(means=means,covs=covs,cross_covs=cross)
    if fixed_q:posterior=jax.lax.stop_gradient(posterior)
    kf,gamma,kp=ConstrainedIVFreeEnergy(RPSSM(p,[])).get_loss_terms(chain,sites,posterior)
    return -(gamma-kf-kp)/(N*T)-j.log(float(N))

def main():
    pins=source_check();rng=np.random.default_rng(5600);x=j.asarray(rng.normal(size=(7,6)))
    native_rows=[];integration_rows=[];difference_rows=[]
    for point in [(.2,.8,.7),(.7,2.,1.1),(-.4,3.,.4),(.95,5.,1.)]:
        theta=j.array([point[0],np.log(point[1]),np.log(point[2])])
        for fixed in [False,True]:
            nv,ng=jax.value_and_grad(native)(theta,x,fixed)
            cv,cg=jax.value_and_grad(lambda z:sample_loss(z,x,x,1.3,'self',fixed,1e-5))(theta)
            errors={'loss':float(abs(nv-cv)),'gradient':float(j.max(j.abs(ng-cg)))}
            assert max(errors.values())<2e-6,(point,fixed,errors,np.asarray(ng),np.asarray(cg))
            native_rows.append({'point':point,'fixed_q':fixed,'errors':errors,'native_gradient':np.asarray(ng).tolist(),'independent_gradient':np.asarray(cg).tolist()})
    for a,r,T in [(.95,.3,6),(.95,3.,24)]:
        S=np.asarray(corr(a,T))+r*np.eye(T);R=np.linalg.cholesky(S)
        points=j.asarray(np.concatenate([np.sqrt(T)*R.T,-np.sqrt(T)*R.T]))
        for theta in [j.array([a,np.log(1/r),0.]),j.array([.35,np.log(2.),np.log(.7)])]:
            for fixed in [False,True]:
                for jitter in [0.,1e-5]:
                    fn=lambda z:population_loss(z,a,r,T,fixed,jitter)
                    other=lambda z:sample_loss(z,points,points,1+r,'integrated',fixed,jitter)
                    pv,pg=jax.value_and_grad(fn)(theta);sv,sg=jax.value_and_grad(other)(theta)
                    error=max(float(abs(pv-sv)),float(j.max(j.abs(pg-sg))))
                    assert error<1e-8,error
                    integration_rows.append({'T':T,'r':r,'theta':np.asarray(theta).tolist(),'fixed_q':fixed,'jitter':jitter,'max_error':error})
                    if not fixed:
                        h=1e-5;z=np.asarray(theta);fd=[]
                        for k in range(3):
                            d=np.eye(3)[k]*h
                            def f(t):return population(t[0],np.exp(t[1]),np.exp(t[2]),a,r,T,jitter)['loss']
                            fd.append((f(z+d)-f(z-d))/(2*h))
                        err=float(np.max(np.abs(np.asarray(pg)-fd)))
                        assert err<3e-6,(z,pg,fd,err)
                        difference_rows.append({'T':T,'r':r,'jitter':jitter,'max_gradient_error':err})
    assert source_check()==pins
    record={'utc':utc(),'status':'PASS_NATIVE_GRADIENTS_AND_EXACT_REFERENCE_EXPECTATION','native':native_rows,'sigma_point_expectation':integration_rows,'central_difference':difference_rows,'source_files_sha256':pins,'protocol_sha256':sha(ROOT/'PROTOCOL.md'),'script_sha256':sha(__file__),'math_sha256':sha(ROOT/'scripts/math56.py'),'new_training_updates':0}
    dump(OUT/'QUALIFICATION56_V1.json',record)
    print(json.dumps({'status':record['status'],'native_gradient_max_error':max(v['errors']['gradient'] for v in native_rows),'sigma_point_max_error':max(v['max_error'] for v in integration_rows),'central_difference_max_error':max(v['max_gradient_error'] for v in difference_rows)}),flush=True)

if __name__=='__main__':main()
