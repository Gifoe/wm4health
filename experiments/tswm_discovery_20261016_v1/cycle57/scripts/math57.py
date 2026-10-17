"""Controlled invariant-rotation Gaussian equations and native auxiliary loss."""
import math
import torch

def rotation(angle):
    c=angle.cos();s=angle.sin()
    return torch.stack([c,-s,s,c],-1).reshape(*angle.shape,2,2)

def rho_from(raw):return .05+.945*raw.sigmoid()
def raw_rho(rho):return math.log((rho-.05)/(.995-rho))

def inv2(A):
    a,b,c,d=A[...,0,0],A[...,0,1],A[...,1,0],A[...,1,1]
    det=a*d-b*c
    return torch.stack([d,-b,-c,a],-1).reshape(A.shape)/det[...,None,None]

def ld2(A):return (A[...,0,0]*A[...,1,1]-A[...,0,1]*A[...,1,0]).log()

def prior(theta,actions):
    B,L=actions.shape;T=L+1;rho=rho_from(theta[0]);q=1-rho.square()
    As=rho*rotation(theta[1:3][actions.long()])
    diagonal=torch.cat([theta.new_ones(1),theta.new_ones(T-2)*(1+rho.square()),theta.new_ones(1)])/q
    J=torch.diag(diagonal.repeat_interleave(2)).expand(B,-1,-1).clone()
    t=torch.arange(T-1,device=theta.device);k=torch.arange(2,device=theta.device)
    row=((t[:,None,None]+1)*2+k[None,:,None]).expand(-1,-1,2).reshape(-1)
    col=(t[:,None,None]*2+k[None,None,:]).expand(-1,2,-1).reshape(-1)
    lower=(-As/q).reshape(B,-1)
    J[:,row,col]=lower;J[:,col,row]=lower
    return J,As,q,(T-1)*2*q.log()

def posterior(theta,lam,u,actions,jitter=1e-5,fixed=False):
    B,T,K=u.shape;J,As,q,prior_ld=prior(theta,actions)
    Jp=J+torch.diag(lam.repeat(T))[None]
    chol=torch.linalg.cholesky(Jp)
    V=torch.cholesky_inverse(chol)
    m=(V@(u*lam).reshape(B,-1,1)).reshape(B,T,K)
    idx=torch.arange(T,device=u.device)[:,None]*2+torch.arange(2,device=u.device)[None]
    C0=V[:,idx[:,:,None],idx[:,None,:]]
    X0=V[:,idx[1:,:,None],idx[:-1,None,:]]
    C=C0+jitter*torch.eye(K,device=u.device,dtype=u.dtype)
    cross=C[:,1:]@(inv2(C0[:,1:])@X0)
    conditional=C[:,1:]-cross@inv2(C[:,:-1])@cross.transpose(-1,-2)
    ld=ld2(C[:,0])+ld2(conditional).sum(1)
    if fixed:m,C,cross,ld=(x.detach() for x in [m,C,cross,ld])
    trace=(J.diagonal(dim1=-2,dim2=-1).reshape(B,T,K)*C.diagonal(dim1=-2,dim2=-1)).sum((1,2))
    trace=trace+2*((-As/q)*cross).sum((1,2,3))
    mm=m.reshape(B,-1)
    kl=.5*(trace+torch.einsum('bi,bij,bj->b',mm,J,mm)-T*K+prior_ld-ld)
    return {'m':m,'C':C,'cross':cross,'kl':kl,'J':J,'chol':chol,'prior_ld':prior_ld}

def kernels(lam,u,ur,post):
    C,m=post['C'],post['m'];P=inv2(C)
    total=P+torch.diag(lam);W=inv2(total)
    h=(P@m[...,None]).squeeze(-1)
    constant=.5*(torch.log1p(lam).sum()-ld2(total)+torch.einsum('bti,btij,btj->bt',h,W,h))
    linear=(W@h[...,None]).squeeze(-1)*lam
    quadratic=.5*(lam[:,None]*W*lam[None,:]-torch.diag(lam.square()/(1+lam)))
    G=constant[:,:,None]+torch.einsum('bti,nti->btn',linear,ur)+torch.einsum('nti,btij,ntj->btn',ur,quadratic,ur)
    positive=constant+(linear*u).sum(-1)+torch.einsum('bti,btij,btj->bt',u,quadratic,u)
    diff=(u-m)*lam
    qf=.5*((C.diagonal(dim1=-2,dim2=-1)*lam).sum(-1)-ld2(total)-ld2(C)+torch.einsum('bti,btij,btj->bt',diff,W,diff))
    return G,positive,qf

def auxiliary(theta,lam,u,ur,actions,log_running=None,jitter=1e-5,fixed=False):
    post=posterior(theta,lam,u,actions,jitter,fixed)
    G,pos,qf=kernels(lam,u,ur,post)
    logmean=torch.logsumexp(G,-1)-math.log(ur.shape[0])
    if log_running is None:denom=logmean
    else:
        weights=torch.exp(G-log_running.detach()[...,None]).detach()
        denom=(weights*G).mean(-1)
    loss=(-pos+denom+qf).mean()+post['kl'].mean()/u.shape[1]
    return loss,post,logmean

def logcosh(x):return x.abs()+torch.log1p(torch.exp(-2*x.abs()))-math.log(2)

def proper_loss(theta,lam,k,x,actions):
    y=torch.sinh(k*x)/k
    B,T,K=x.shape;post=posterior(theta,lam,y,actions,jitter=0.)
    chol=post['chol'];h=(y*lam).reshape(B,-1)
    ld=2*chol.diagonal(dim1=-2,dim2=-1).log().sum(-1)
    quad=(y.square()*lam).sum((1,2))-(h*post['m'].reshape(B,-1)).sum(-1)
    nll=.5*(T*K*math.log(2*math.pi)+post['prior_ld']-T*lam.log().sum()+ld+quad)
    return (nll-logcosh(k*x).sum((1,2))).mean()/T

def update_running(old,logmean,gamma):
    new=torch.logaddexp(old+math.log1p(-gamma),logmean.detach()+math.log(gamma)) if gamma<1 else logmean.detach()
    return torch.where(torch.isfinite(old),new,logmean.detach())
