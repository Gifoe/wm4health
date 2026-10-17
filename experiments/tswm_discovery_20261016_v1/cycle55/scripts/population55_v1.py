"""Scalar Gaussian expectation of the pinned RP auxiliary loss, not a new loss."""
import numpy as np
from scipy.special import logsumexp

def correlation(a, T):
    k=np.arange(T)
    return a**np.abs(k[:,None]-k[None,:])

def precision(a,T):
    assert T>=3 and abs(a)<1
    Q=1-a*a
    J=np.diag(np.r_[1.,np.full(T-2,1+a*a),1.])/Q
    J+=np.diag(np.full(T-1,-a/Q),1)+np.diag(np.full(T-1,-a/Q),-1)
    return J

def posterior_terms(a,lam,b,T,jitter):
    J=precision(a,T);V=np.linalg.solve(J+lam*np.eye(T),np.eye(T))
    C=np.diag(V).copy()+jitter;cross=np.diag(V,1).copy();L=lam*b*V
    conditionals=C[1:]-cross**2/C[:-1]
    assert np.all(conditionals>0)
    logdet_q=float(np.log(C[0])+np.log(conditionals).sum())
    return J,V,C,cross,L,logdet_q

def population(a,lam,b,a_star,r,T,jitter=0.,frozen_q=None):
    assert lam>0 and b>=0 and abs(a)<1 and r>0
    S=correlation(a_star,T)+r*np.eye(T);v=1+r
    J=precision(a,T)
    qa,ql,qb=(a,lam,b) if frozen_q is None else frozen_q
    _,V,C,cross,L,logdet_q=posterior_terms(qa,ql,qb,T,jitter)
    M=L@S@L.T;mx=np.diag(L@S)
    delta=L-b*np.eye(T);difference=np.diag(delta@S@delta.T)
    coefficient=lam/(1+lam*C)
    quadratic=.5*lam*lam*(C-1)/((1+lam*C)*(1+lam))
    factor_variance=b*b*v;denominator=1-2*quadratic*factor_variance
    if np.any(denominator<=0):return {'loss':np.inf,'normalizer_exists':False}
    gamma=coefficient*b*mx+quadratic*factor_variance+.5*np.log(denominator)-.5*coefficient**2*np.diag(M)*factor_variance/denominator
    kl_factor=.5*(lam*C-np.log1p(lam*C)+lam*lam/(lam+1/C)*difference)
    trace_cov=float(np.dot(np.diag(J),C)+2*np.dot(np.diag(J,1),cross))
    kl_prior=.5*(trace_cov+np.trace(J@M)-T+(T-1)*np.log1p(-a*a)-logdet_q)
    loss=float((-gamma.sum()+kl_factor.sum()+kl_prior)/T)
    R_eff=1/(1/v+(lam*b)**2/(1+lam));C_eff=R_eff*lam*b
    model_cov=C_eff*C_eff*correlation(a,T)+R_eff*np.eye(T)
    _,ld=np.linalg.slogdet(model_cov);_,true_ld=np.linalg.slogdet(S)
    proper=float(.5*(ld+np.trace(np.linalg.solve(model_cov,S))-T-T*np.log(v))/T)
    true_optimum=float(.5*(true_ld-T*np.log(v))/T)
    reference_variance=1/(1+lam)+(lam*b/(1+lam))**2*v
    return {'loss':loss,'normalizer_exists':True,'proper_likelihood_loss':proper,'bound_gap':loss-proper,'true_likelihood_optimum':true_optimum,'reference_variance':float(reference_variance),'kl_factor':float(kl_factor.sum()/T),'kl_prior':float(kl_prior/T),'mean_log_gamma':float(gamma.mean()),'effective_emission_slope':float(C_eff),'effective_emission_variance':float(R_eff)}

def finite(a,lam,b,x,jitter=1e-5):
    N,T=x.shape;J,V,C,cross,L,logdet_q=posterior_terms(a,lam,b,T,jitter)
    means=x@L.T;u=b*x
    qf=.5*(lam*C-np.log1p(lam*C)+lam*lam/(lam+1/C)*(means-u)**2)
    trace_cov=np.dot(np.diag(J),C)+2*np.dot(np.diag(J,1),cross)
    qp=.5*(trace_cov+np.einsum('nt,tu,nu->n',means,J,means)-T+(T-1)*np.log1p(-a*a)-logdet_q)
    G=np.empty((T,N,N))
    for t in range(T):
        P=1/C[t]
        G[t]=.5*np.log((lam+1)/(lam+P))+.5*((lam*u[None,:,t]+P*means[:,None,t])**2/(lam+P)-(lam*u[None,:,t])**2/(lam+1))
    gamma=np.diagonal(G,axis1=-2,axis2=-1)-logsumexp(G,axis=-1)
    loss=float((-gamma.sum()+qf.sum()+qp.sum())/(N*T))
    return {'loss':loss,'centered_loss':loss-np.log(N),'means':means,'covariances':C,'cross_covariances':cross,'kl_factor':float(qf.mean()),'kl_prior':float(qp.mean()/T),'mean_log_gamma':float(gamma.mean())}

def sigma_point_population(a,lam,b,a_star,r,T,jitter=0.):
    S=correlation(a_star,T)+r*np.eye(T);root=np.linalg.cholesky(S)
    points=np.concatenate([np.sqrt(T)*root.T,-np.sqrt(T)*root.T])
    J,V,C,cross,L,logdet_q=posterior_terms(a,lam,b,T,jitter)
    m=points@L.T;u=b*points;v=1+r
    coef=lam/(1+lam*C);quad=.5*lam*lam*(C-1)/((1+lam*C)*(1+lam));tau=b*b*v;D=1-2*quad*tau
    gamma=coef*m*u+quad*u*u+.5*np.log(D)-.5*coef**2*m*m*tau/D
    qf=.5*(lam*C-np.log1p(lam*C)+lam*lam/(lam+1/C)*(m-u)**2)
    qp=.5*(np.dot(np.diag(J),C)+2*np.dot(np.diag(J,1),cross)+np.einsum('nt,tu,nu->n',m,J,m)-T+(T-1)*np.log1p(-a*a)-logdet_q)
    return float((-gamma.sum(1)+qf.sum(1)+qp).mean()/T)
