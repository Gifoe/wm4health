"""Native heteroscedastic decoder and proper physical distribution scores."""
import runtime_profile
import hashlib,json,pathlib,sys,math
ROOT=pathlib.Path(__file__).resolve().parents[1];PREV=ROOT.parent/'cycle23'
sys.path.insert(0,str(PREV/'scripts'))
from native_context_runtime import load,verify_source
import numpy as np
import torch
DEVICE='cuda:0';M=1024
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def tensor(x):return torch.as_tensor(x,dtype=torch.float32,device=DEVICE)
def decode(model,beta,state,action):
    shape=beta.shape[:-1];b=beta.reshape(-1,16).float();x=torch.broadcast_to(state,(*shape,3)).reshape(-1,3);u=torch.broadcast_to(action,(*shape,1)).reshape(-1,1);mus=[];vs=[]
    for t in range(0,len(b),16384):
        d=model(x[t:t+16384].contiguous(),u[t:t+16384].contiguous(),b[t:t+16384].contiguous());mus.append(d.mean);vs.append(d.variance)
    mu=torch.cat(mus).reshape(*shape,3);v=torch.cat(vs).reshape(*shape,3)
    assert torch.isfinite(mu).all() and torch.isfinite(v).all() and (v>0).all()
    return mu.double(),v.double()
def query_decode(model,beta,theta):
    means=[];vs=[]
    for th in theta:
        x=tensor([math.cos(float(th)),math.sin(float(th)),0.]);mu,v=decode(model,beta,x,tensor([1.5]));means.append(mu[...,2]);vs.append(v[...,2])
    return torch.stack(means,-1),torch.stack(vs,-1)
def encode(encoder,states,actions,n):
    if not n:
        q=encoder.empty_set_context();return q.loc.expand(states.shape[1],16).double(),q.scale.expand(states.shape[1],16).double()
    q=encoder.forward_tensor(states[:n],actions[:n],states[1:n+1]);return q.loc.double(),q.scale.double()
def append(encoder,states,actions,n,px,u,py):
    x=px.expand(1,states.shape[1],3);a=torch.full((1,states.shape[1],1),u,device=DEVICE);y=py[None]
    q=encoder.forward_tensor(x,a,y) if not n else encoder.forward_broadcast(states[:n],actions[:n],states[1:n+1],x,a,y)
    return q.loc.double(),q.scale.double()
def lognormal(b,loc,scale):return (-.5*((b-loc[:,None])/scale[:,None])**2-torch.log(scale[:,None])-.5*math.log(2*math.pi)).sum(-1)
def absolute_normal(d,s):
    z=d/s;return s*math.sqrt(2/math.pi)*torch.exp(-.5*z*z)+d*(2*torch.special.ndtr(z)-1)
def scores(mu,v,mu2,v2,truth,noise,noise2):
    # Independent paired components, not the same beta twice. Native noise is heteroscedastic.
    y=truth[:,None,:];mean=mu.mean(1);second=(mu*mu+v).mean(1)
    crps=absolute_normal(mu-y,torch.sqrt(v)).mean(1)-.5*absolute_normal(mu-mu2,torch.sqrt(v+v2)).mean(1)
    pit=torch.special.ndtr((y-mu)/torch.sqrt(v)).mean(1)
    samples=(mu+torch.sqrt(v)*noise)/8.;samples2=(mu2+torch.sqrt(v2)*noise2)/8.
    es=torch.linalg.vector_norm(samples-truth[:,None,:]/8.,dim=-1).mean(1)-.5*torch.linalg.vector_norm(samples-samples2,dim=-1).mean(1)
    variogram=torch.zeros(len(mu),device=DEVICE,dtype=torch.float64)
    for i in range(4):
        for j in range(i):
            pred=absolute_normal((mu[...,i]-mu[...,j])/8.,torch.sqrt(v[...,i]+v[...,j])/8.).mean(1)
            variogram+=(pred-abs(truth[:,i]-truth[:,j])/8.)**2/6.
    centered=mu-mean[:,None];cov=torch.einsum('nmi,nmj->nij',centered,centered)/mu.shape[1]+torch.diag_embed(v.mean(1))
    return dict(crps=crps,pit=pit,mean=mean,variance=second-mean**2,covariance=cov,energy=es,variogram=variogram,squared_error=(mean-truth)**2)
def summary(s):return dict(crps=float(s['crps'].mean()),query_crps=s['crps'].mean(0).tolist(),energy=float(s['energy'].mean()),variogram=float(s['variogram'].mean()),rmse=float(s['squared_error'].mean().sqrt()),query_rmse=s['squared_error'].mean(0).sqrt().tolist(),coverage95=float(((s['pit']>=.025)&(s['pit']<=.975)).double().mean()),query_coverage95=((s['pit']>=.025)&(s['pit']<=.975)).double().mean(0).tolist(),predictive_variance=float(s['variance'].mean()))
def gaussian_samples(loc,cov,z):
    if cov.ndim==2:return loc[:,None]+torch.sqrt(cov[:,None])*z
    e,V=torch.linalg.eigh(cov);assert e.min()>-1e-7
    return loc[:,None]+torch.einsum('nij,nmj->nmi',V*torch.sqrt(e.clamp_min(0))[:,None,:],z)
def slr(model,loc,scale,px,u,py,z):
    beta=loc[:,None]+scale[:,None]*z;mu,v=decode(model,beta,px,tensor([u]));db=beta-beta.mean(1,keepdim=True);dm=mu-mu.mean(1,keepdim=True)
    cb=torch.einsum('nmi,nmj->nij',db,db)/z.shape[1];cross=torch.einsum('nmi,nmj->nij',db,dm)/z.shape[1]
    # Regress h(beta); residual covariance plus native heteroscedastic noise is PSD.
    H=torch.linalg.solve(cb,cross).transpose(-1,-2);res=dm-torch.einsum('nsi,nmi->nms',H,db)
    omega=torch.einsum('nmi,nmj->nij',res,res)/z.shape[1]+torch.diag_embed(v.mean(1))
    P=torch.diag_embed(scale**2);PH=P@H.transpose(-1,-2);S=H@PH+omega;K=torch.linalg.solve(S,PH.transpose(-1,-2)).transpose(-1,-2)
    outloc=loc+torch.einsum('nbs,ns->nb',K,py.double()-mu.mean(1));outcov=P-K@H@P;outcov=(outcov+outcov.transpose(-1,-2))*.5
    assert torch.linalg.eigvalsh(outcov).min()>-1e-7
    return outloc,outcov
def importance(model,loc,scale,postloc,postscale,px,u,py,count,z):
    half=count//2;b=torch.cat((loc[:,None]+scale[:,None]*z[:,:half],postloc[:,None]+postscale[:,None]*z[:,half:count]),1)
    lp=lognormal(b,loc,scale);lq=torch.logaddexp(lp,lognormal(b,postloc,postscale))-math.log(2)
    mu,v=decode(model,b,px,tensor([u]));ll=(-.5*(torch.log(2*math.pi*v)+(py[:,None].double()-mu)**2/v)).sum(-1)
    w=torch.softmax(lp+ll-lq,1);m=(w[...,None]*b).sum(1);d=b-m[:,None];cov=torch.einsum('nm,nmi,nmj->nij',w,d,d);cov=(cov+cov.transpose(-1,-2))*.5
    assert torch.isfinite(w).all() and torch.isfinite(cov).all() and torch.linalg.eigvalsh(cov).min()>-1e-7
    return b,w,m,cov,1/(w*w).sum(1)
