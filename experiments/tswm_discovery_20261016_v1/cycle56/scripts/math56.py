"""Native-matched scalar finite and integrated-reference losses in JAX."""
import jax
import jax.numpy as j
from jax.scipy.special import logsumexp

def corr(a,T):
    k=j.arange(T)
    return a**j.abs(k[:,None]-k[None,:])

def prec(a,T):
    q=1-a*a
    return (j.diag(j.concatenate((j.ones(1),j.full((T-2,),1+a*a),j.ones(1))))
            +j.diag(j.full((T-1,),-a),1)+j.diag(j.full((T-1,),-a),-1))/q

def qterms(theta,T,jitter,fixed_q):
    a=theta[0];lam=j.exp(theta[1]);b=j.exp(theta[2])
    V=j.linalg.solve(prec(a,T)+lam*j.eye(T),j.eye(T))
    C=j.diag(V)+jitter
    def step(filtered,_):
        predicted=a*a*filtered+1-a*a
        return predicted/(1+lam*predicted),a*filtered/predicted
    _,gains=jax.lax.scan(step,1/(1+lam),None,length=T-1)
    cross=j.diag(V,1)+jitter*gains
    L=lam*b*V
    ld=j.log(C[0])+j.log(C[1:]-cross**2/C[:-1]).sum()
    terms=(C,cross,L,ld)
    return jax.tree_util.tree_map(jax.lax.stop_gradient,terms) if fixed_q else terms

def sample_loss(theta,x,reference,variance,mode='self',fixed_q=False,jitter=1e-5):
    M,T=x.shape;B=reference.shape[0]
    a=theta[0];lam=j.exp(theta[1]);b=j.exp(theta[2]);J=prec(a,T)
    C,cross,L,ld=qterms(theta,T,jitter,fixed_q)
    means=x@L.T;u=b*x
    coef=lam/(1+lam*C)
    quad=.5*lam**2*(C-1)/((1+lam*C)*(1+lam))
    positive=coef*means*u+quad*u*u
    if mode=='integrated':
        tau=b*b*variance;D=1-2*quad*tau
        logden=-.5*j.log(D)+.5*coef**2*means**2*tau/D
    else:
        ur=b*reference
        H=(coef[None,:,None]*means[:,:,None]*ur.T[None,:,:]
           +quad[None,:,None]*ur.T[None,:,:]**2)
        count=B
        if mode=='loo':
            mask=j.arange(M)[:,None,None]==j.arange(B)[None,None,:]
            H=j.where(mask,-j.inf,H);count=B-1
        logden=logsumexp(H,axis=-1)-j.log(float(count))
    gamma=positive-logden
    qf=.5*(lam*C-j.log1p(lam*C)+lam**2/(lam+1/C)*(means-u)**2)
    tc=j.dot(j.diag(J),C)+2*j.dot(j.diag(J,1),cross)
    qp=.5*(tc+j.einsum('nt,tu,nu->n',means,J,means)-T+(T-1)*j.log1p(-a*a)-ld)
    return (-gamma.sum()+qf.sum()+qp.sum())/(M*T)

def population_loss(theta,a_star,r,T,fixed_q=False,jitter=1e-5):
    a=theta[0];lam=j.exp(theta[1]);b=j.exp(theta[2]);J=prec(a,T)
    S=corr(a_star,T)+r*j.eye(T);v=1+r
    C,cross,L,ld=qterms(theta,T,jitter,fixed_q)
    M=L@S@L.T;mx=j.diag(L@S);delta=L-b*j.eye(T)
    diff=j.diag(delta@S@delta.T)
    coef=lam/(1+lam*C);quad=.5*lam**2*(C-1)/((1+lam*C)*(1+lam))
    tau=b*b*v;D=1-2*quad*tau
    gamma=coef*b*mx+quad*tau+.5*j.log(D)-.5*coef**2*j.diag(M)*tau/D
    qf=.5*(lam*C-j.log1p(lam*C)+lam**2/(lam+1/C)*diff)
    tc=j.dot(j.diag(J),C)+2*j.dot(j.diag(J,1),cross)
    qp=.5*(tc+j.trace(J@M)-T+(T-1)*j.log1p(-a*a)-ld)
    return (-gamma.sum()+qf.sum()+qp)/T
