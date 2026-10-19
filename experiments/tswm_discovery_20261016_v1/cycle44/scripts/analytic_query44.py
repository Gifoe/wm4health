"""Ordinary Gaussian closure and conditional observation-noise integration."""
from common44 import *

@torch.no_grad()
def ssm_query_moments(model,history):
 history=torch.as_tensor(history,device=model.U.device,dtype=model.U.dtype);mean,cov,_=model.filter(history);assert len(mean)==1
 A,Q,_,r=model.matrices();loading=model.U*model.scale;means=[];covariances=[];m=mean[0];p=cov
 for _ in range(FUT):m=A@m;p=A@p@A.T+Q;means.append(m);covariances.append(p)
 means=torch.stack(means);field_means=means@loading.T;field_variance=torch.stack([(loading@p0*loading).sum(1)+r for p0 in covariances]);row=loading.mean(0);spatial_cov=torch.empty(FUT,FUT,device=m.device,dtype=m.dtype)
 for j in range(FUT):
  cross=covariances[j]
  for i in range(j,FUT):
   if i>j:cross=A@cross
   value=row@cross@row
   if i==j:value=value+r/P
   spatial_cov[i,j]=value;spatial_cov[j,i]=value
 gram=loading.T@loading;rough=[]
 for t in range(1,FUT):
  delta=field_means[t]-field_means[t-1];pcov=covariances[t]+covariances[t-1]-A@covariances[t-1]-covariances[t-1]@A.T;rough.append((delta.square().sum()+torch.trace(gram@pcov)+2*r*P)/P)
 return {'field_mean':field_means,'field_variance':field_variance,'spatial_mean':field_means.mean(1),'spatial_covariance':(spatial_cov+spatial_cov.T)/2,'mean':field_means.mean(),'roughness':torch.stack(rough).mean(),'posterior_mean':mean[0],'posterior_covariance':cov,'A':A,'Q':Q,'R':r,'loading':loading}

@torch.no_grad()
def ssm_analytic_readout(moments,spec):
 occupancy=torch.special.ndtr((moments['field_mean']-spec['point_threshold'])/moments['field_variance'].sqrt()).mean();return {0:moments['mean']/spec['scales'][0],2:moments['roughness']/spec['scales'][2],3:occupancy/spec['scales'][3]}

@torch.no_grad()
def ar_conditional_readout(model,pc_paths,spec,normal_noise=None):
 # pc_paths [batch, 16, rank], sampled predicted observed coordinates.
 ymean=(pc_paths*model.scale)@model.U.T;spatial=ymean.mean(-1);d=model.U.shape[0];eye=torch.eye(d,device=model.U.device,dtype=model.U.dtype);transform=eye-model.U@model.U.T;cov=model.tail_noise*(transform@transform.T);diagonal=cov.diagonal().clamp_min(1e-30);spatial_variance=model.tail_noise*transform.mean(0).square().sum();spatial_sd=spatial_variance.clamp_min(1e-30).sqrt()
 mean=spatial.mean(-1);rough=torch.diff(ymean,dim=-2).square().mean((-1,-2))+2*torch.trace(cov)/P;occupancy=torch.special.ndtr((ymean-spec['point_threshold'])/diagonal.sqrt()).mean((-1,-2));below=torch.special.ndtr((spec['spatial_mean_threshold']-spatial)/spatial_sd);crossing=1-torch.prod(below,dim=-1)
 if normal_noise is None:normal_noise=torch.zeros_like(spatial)
 marginal_means=spatial+spatial_sd*normal_noise;tau=spec['smoothmax_tau'];smooth=tau*(torch.logsumexp(marginal_means/tau,-1)-math.log(FUT));scales=torch.tensor(spec['scales'],device=pc_paths.device,dtype=pc_paths.dtype)
 return torch.stack((mean,smooth,rough,occupancy,crossing),-1)/scales

@torch.no_grad()
def draw_ar_pc(model,history,n,draw_seed,antithetic=False):
 model.eval();hist=torch.as_tensor(history,device=model.U.device,dtype=model.U.dtype);h0,hidden=model.gru(model.project(hist));features=h0[:,-1].repeat_interleave(n,0);hidden=hidden.repeat_interleave(n,1);b=len(features);g=torch.Generator(device=model.U.device).manual_seed(draw_seed);paths=[];scalar_noise=[]
 for t in range(FUT):
  logits,means,L=model.params(features);u=torch.rand(b,1,device=model.U.device,generator=g);eps=torch.randn(b,model.d,device=model.U.device,generator=g);tail=torch.randn(b,device=model.U.device,generator=g)
  if antithetic:u=1-u;eps=-eps;tail=-tail
  k=(u>logits.softmax(-1).cumsum(-1)).sum(-1).clamp_max(model.K-1);j=torch.arange(b,device=model.U.device);pc=means[j,k]+torch.bmm(L[j,k],eps[:,:,None]).squeeze(-1);paths.append(pc);scalar_noise.append(tail);out,hidden=model.gru(pc[:,None],hidden);features=out[:,0]
 return torch.stack(paths,1),torch.stack(scalar_noise,1)
