"""Unmodified JAX source supplies cross-runtime loss, moment and AD fixtures."""
from common57 import *
os.environ['JAX_PLATFORM_NAME']='cpu';os.environ['JAX_ENABLE_X64']='true'
import numpy as np
import jax,jax.numpy as j
from rp_ssm.dists import GaussianNatParam,LGChainDistParam
from rp_ssm.rpm import ConstrainedIVFreeEnergy
from rp_ssm.dists_utils import transitions_to_marginals,parallel_smoother

def native(theta,x,actions,fixed):
    N,T,K=x.shape;rho=.05+.945*jax.nn.sigmoid(theta[0]);lam=j.exp(theta[3:5])
    angle=theta[1:3][actions];c=j.cos(angle);s=j.sin(angle)
    As=rho*j.stack([c,-s,s,c],-1).reshape(N,T-1,2,2)
    u=x@theta[5:9].reshape(2,2).T+theta[9:11]
    def one(matrices):
        params={'A':rho*j.eye(2),'b':j.zeros(2),'m1':j.zeros(2),'Q1':j.eye(2),'Q':(1-rho*rho)*j.eye(2),'As':matrices,'bs':j.zeros((T-1,2))}
        return transitions_to_marginals(params,T,False),params
    chain,inf=jax.vmap(one)(As)
    sites=GaussianNatParam(p=j.broadcast_to(j.diag(lam),(1,N,T,K,K)),pwm=(u*lam)[None])
    local=GaussianNatParam(p=sites.params['p'][0],pwm=sites.params['pwm'][0]).dist_param
    m,C,X=jax.vmap(lambda f,p:parallel_smoother(p,f,2))(local,inf)
    posterior=LGChainDistParam(means=m,covs=C,cross_covs=X)
    if fixed:posterior=jax.lax.stop_gradient(posterior)
    kf,gamma,kp=ConstrainedIVFreeEnergy(None).get_loss_terms(chain,sites,posterior)
    return -(gamma-kf-kp)/(N*T)-j.log(float(N)),(m,C,X)

def main():
    pins=source_check();rng=np.random.default_rng(5701);x=rng.normal(size=(7,6,2));actions=rng.integers(0,2,size=(7,5))
    points=[]
    for rho,angles,lam,W,b in [(.6,[-.2,.4],[.8,1.7],[[.7,.1],[-.1,1.2]],[.2,-.1]),(.94,[-.3,.2],[3.,2.],[[.8,.05],[.1,.6]],[0.,.1])]:
        theta=np.r_[np.log((rho-.05)/(.995-rho)),angles,np.log(lam),np.asarray(W).reshape(-1),b]
        for fixed in [False,True]:
            (value,moments),grad=jax.value_and_grad(native,has_aux=True)(j.asarray(theta),j.asarray(x),j.asarray(actions),fixed)
            assert np.isfinite(np.asarray(grad)).all()
            points.append({'theta':theta.tolist(),'fixed_q':fixed,'loss':float(value),'gradient':np.asarray(grad).tolist(),'moments':{k:np.asarray(v).tolist() for k,v in zip(['mean','cov','cross'],moments)}})
    assert source_check()==pins
    record={'utc':utc(),'status':'NATIVE_CONTROLLED_GAUSSIAN_FIXTURES_READY','source_files_sha256':pins,'x':x.tolist(),'actions':actions.tolist(),'points':points,'protocol_sha256':sha(ROOT/'PROTOCOL.md'),'script_sha256':sha(__file__),'formal_updates':0}
    dump(OUT/'NATIVE_FIXTURE57_V1.json',record)
    print(json.dumps({'status':record['status'],'cases':len(points),'native_loss_calls':'AD value-and-gradient, no optimization'}),flush=True)

if __name__=='__main__':main()
