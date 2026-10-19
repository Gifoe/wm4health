"""Physical-distribution calculations; no training or test-based choices."""
import runtime_profile
import json,pathlib,sys,hashlib,math
ROOT=pathlib.Path(__file__).resolve().parents[1];PREV=ROOT.parent/'cycle23'
sys.path.insert(0,str(PREV/'scripts'))
from native_context_runtime import load,verify_source
import numpy as np
import torch
DEVICE='cuda:0'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def tensor(x):return torch.as_tensor(x,dtype=torch.float32,device=DEVICE)
def normal_grid(k=8193,limit=7.):
    z=torch.linspace(-limit,limit,k,dtype=torch.float64,device=DEVICE)
    w=torch.exp(-z*z/2);w[[0,-1]]*=.5;w/=w.sum();return z,w
def hermite(k=128):
    z,w=np.polynomial.hermite.hermgauss(k)
    return torch.tensor(z*np.sqrt(2),device=DEVICE),torch.tensor(w/np.sqrt(np.pi),device=DEVICE)
def decode(model,beta,u,chunk=16384):
    shape=beta.shape;flat=beta.reshape(-1,1).float();out=[];variance=None
    for start in range(0,len(flat),chunk):
        b=flat[start:start+chunk].contiguous();x=torch.zeros(len(b),2,device=DEVICE);a=torch.full((len(b),1),float(u),device=DEVICE)
        dist=model(x,a,b);mu=dist.mean;var=dist.variance
        assert torch.isfinite(mu).all() and torch.isfinite(var).all() and (var>0).all()
        if variance is None:variance=var[0].clone()
        assert torch.equal(var,variance.expand_as(var)),'Only the declared native constant-noise toy models are supported.'
        out.append(mu)
    return torch.cat(out).reshape(*shape,2),variance.double()
def normal_components(model,loc,scale,u,z):
    beta=loc.double()[...,None]+scale.double()[...,None]*z
    mu,var=decode(model,beta,u);return mu.double(),var,beta
def moments(mu,var,w):
    if w.ndim==1:w=w.expand(mu.shape[:-1])
    first=(mu*w[...,None]).sum(-2);second=((mu*mu+var)*w[...,None]).sum(-2)
    return first,second
def physical_moments(model,loc,scale,u,k=128):
    z,w=hermite(k);mu,var,_=normal_components(model,loc,scale,u,z);a,b=moments(mu,var,w);return a,b
def encode(encoder,states,actions,n):
    if n==0:
        q=encoder.empty_set_context();N=states.shape[1]
        return q.loc.expand(N,1).clone(),q.scale.expand(N,1).clone()
    q=encoder.forward_tensor(states[:n],actions[:n],states[1:n+1]);return q.loc,q.scale
def append_evidence(encoder,states,actions,n,probe,u):
    # probe[N,M,2] or[N,2]; native arbitrary batch dimensions retained.
    if probe.ndim==2:probe=probe[:,None,:]
    N,M,_=probe.shape;x=torch.zeros(1,N,M,2,device=DEVICE);a=torch.full((1,N,M,1),float(u),device=DEVICE);y=probe[None]
    if n==0:q=encoder.forward_tensor(x,a,y)
    else:q=encoder.forward_broadcast(states[:n,:,None,:],actions[:n,:,None,:],states[1:n+1,:,None,:],x,a,y)
    return q.loc[...,0],q.scale[...,0]
def likelihood(mu,var,y):
    return -.5*(torch.log(2*math.pi*var)+(y[...,None,:]-mu)**2/var).sum(-1)
def posterior_weights(logprior,mu,var,y):
    return torch.softmax(logprior+likelihood(mu,var,y),dim=-1)
def score(mu,var,w,y):
    # Full joint Gaussian mixture score, not sum of marginal log scores.
    if w.ndim==1:w=w.expand(mu.shape[:-1])
    ll=likelihood(mu,var,y);logdensity=torch.logsumexp(torch.log(w)+ll,dim=-1)
    first,second=moments(mu,var,w)
    pit=(torch.special.ndtr((y[:,None,:]-mu)/torch.sqrt(var))*w[...,None]).sum(-2)
    return dict(nll=-logdensity,mean=first,second=second,pit=pit,error=first-y)
def summary_scores(s):
    return dict(nll=float(s['nll'].mean()),rmse=float(torch.sqrt((s['error']**2).mean())),
        channel_rmse=torch.sqrt((s['error']**2).mean(0)).cpu().tolist(),
        marginal_coverage95=float(((s['pit']>=.025)&(s['pit']<=.975)).double().mean()),
        channel_coverage95=((s['pit']>=.025)&(s['pit']<=.975)).double().mean(0).cpu().tolist(),
        predictive_variance=float((s['second']-s['mean']**2).mean()))
def summarize_drift(prior_first,prior_second,post_first,post_second):
    # Focus first physical coordinate; second coordinate retained in raw data.
    variance=(prior_second-prior_first**2).clamp_min(1e-15)
    result={}
    for key,prior,post,norm in [('first',prior_first,post_first,torch.sqrt(variance)),('second',prior_second,post_second,prior_second.abs().clamp_min(1e-12))]:
        delta=post.mean(1)-prior;se=post.std(1,unbiased=True)/math.sqrt(post.shape[1])
        relative=delta/norm;noise_corrected=(delta**2-se**2)/norm**2
        out=dict(median_abs_relative=float(relative[:,0].abs().median()),mean_signed_relative=float(relative[:,0].mean()),
            mean_noise_corrected_squared_relative=float(noise_corrected[:,0].mean()),
            fraction_beyond3_mcse=float((delta[:,0].abs()>3*se[:,0]+1e-8).double().mean()),
            max_abs_relative=float(relative[:,0].abs().max()))
        rng=np.random.default_rng(20262490);a=noise_corrected[:,0].cpu().numpy();ids=rng.integers(0,len(a),size=(2000,len(a)))
        out['context_bootstrap95_noise_corrected_squared_relative']=np.quantile(a[ids].mean(1),[.025,.975]).tolist();result[key]=out
    return result
def alpha_weights(states,actions,n,kind,k=8193,probe=None,probe_u=None):
    alpha=torch.linspace(-1,1,k,dtype=torch.float64,device=DEVICE);N=states.shape[1]
    logw=torch.zeros(N,k,dtype=torch.float64,device=DEVICE)
    logw[:,[0,-1]]-=math.log(2)
    # Complete observed prefixes and alpha-independent x0 justify this likelihood.
    A=torch.tensor([[.8,.2],[-.2,.8]],dtype=torch.float64,device=DEVICE)
    for t in range(n):
        g=torch.relu(1-actions[t,:,0].abs()) if kind==1 else torch.relu(actions[t,:,0].abs()-1)
        residual=states[t+1,:,0].double()-(states[t].double()@A.T)[:,0]
        logw-=.5*((residual[:,None]-g[:,None]*alpha)/.01)**2
    if probe is not None:
        g=max(0.,1-abs(probe_u)) if kind==1 else max(0.,abs(probe_u)-1)
        logw-=.5*((probe[:,0,None].double()-g*alpha)/.01)**2
    return alpha,torch.softmax(logw,dim=-1)
def alpha_score(alpha,w,query_u,kind,y):
    g=max(0.,1-abs(query_u)) if kind==1 else max(0.,abs(query_u)-1)
    mu=torch.stack((g*alpha,torch.zeros_like(alpha)),dim=-1)[None].expand(len(y),-1,-1)
    return score(mu,torch.full((2,),.0001,dtype=torch.float64,device=DEVICE),w,y)
