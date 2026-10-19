"""Read-only selected world laws, lawful coupling, ordinary prefix caches."""
from common44 import *
from contextlib import contextmanager
from analytic_query44 import ssm_query_moments,ssm_analytic_readout,ar_conditional_readout,draw_ar_pc

def function_values(paths,spec,coarse=False):
 # Return normalized functional units, with the exact constant-path proxy.
 scale=torch.tensor(spec['scales'],device=paths.device,dtype=paths.dtype)
 if coarse:
  field=paths[...,0,:];m=field.mean(-1);raw=torch.stack((m,m,torch.zeros_like(m),(field>spec['point_threshold']).to(paths.dtype).mean(-1),(m>spec['spatial_mean_threshold']).to(paths.dtype)),-1)
 else:
  m=paths.mean(-1);tau=spec['smoothmax_tau'];raw=torch.stack((m.mean(-1),tau*(torch.logsumexp(m/tau,-1)-math.log(FUT)),torch.diff(paths,dim=-2).square().mean((-1,-2)),(paths>spec['point_threshold']).to(paths.dtype).mean((-1,-2)),(m.max(-1).values>spec['spatial_mean_threshold']).to(paths.dtype)),-1)
 return raw/scale

def selected_model(summary):
 case=summary['case'];family=summary['family'];data=R/'data'/case;pca=dict(np.load(data/'pca.npz'));checkpoint=R/'outputs/fits_v1'/f"{case}_{family}_{summary['seed']}"/'model.pt';assert sha(checkpoint)==summary['checkpoint_sha256'];saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
 if family.startswith('spf'):model=make_spf(family=='spf_condition_trainable');module=model.model
 elif family=='ar_mixture':model=AR(pca).cuda();module=model
 else:model=LGSSM(pca,np.load(data/'train.npz')['observations']).cuda();module=model
 module.load_state_dict(saved['selected_state']);module.eval();return model,module

@contextmanager
def prefix_cache(family,model):
 # Cache only deterministic results for one immutable history. No law changes.
 if family.startswith('spf'):
  module=model.model.cond_pos_embed;original=module.forward;cache={}
  def cached(value):
   def shapes(x):
    if torch.is_tensor(x):return tuple(x.shape)
    return tuple(shapes(v) for v in x)
   key=shapes(value)
   if key not in cache:
    with torch.no_grad():cache[key]=original(value)
   return cache[key]
  module.forward=cached
  try:yield
  finally:module.forward=original
 elif family=='ar_mixture':
  original=model.gru.forward;cache={}
  def cached(x,hx=None):
   if hx is None and x.shape[0]==1 and x.shape[1]==H:
    if not cache:
     with torch.no_grad():cache['value']=original(x,hx)
    return cache['value']
   return original(x,hx)
  model.gru.forward=cached
  try:yield
  finally:model.gru.forward=original
 else:
  original=model.filter;cache={}
  def cached(y,with_parts=False):
   if not with_parts and y.shape[0]==1 and y.shape[1]==H:
    if not cache:
     with torch.no_grad():cache['value']=original(y,with_parts)
    return cache['value']
   return original(y,with_parts)
  model.filter=cached
  try:yield
  finally:model.filter=original

@torch.no_grad()
def warm_prefix(family,model,history,n=64):
 if family.startswith('spf'):
  hist=torch.as_tensor(history,device=model.device,dtype=torch.float32).repeat_interleave(n,0);xs,_=native_inputs(hist,model.device)
  for key in ['decadal','monthly']:model.model.cond_pos_embed([[xs[key].transpose(1,2).contiguous()]])
 elif family=='ar_mixture':model.gru(model.project(torch.as_tensor(history,device=model.U.device,dtype=model.U.dtype)))
 else:model.filter(torch.as_tensor(history,device=model.U.device,dtype=model.U.dtype))

@torch.no_grad()
def ssm_marginal_readout(moments,analytic,spec,n,draw_seed,antithetic=False):
 mean=moments['spatial_mean'];L=torch.linalg.cholesky(moments['spatial_covariance']);g=torch.Generator(device=mean.device).manual_seed(draw_seed);noise=torch.randn(n,FUT,device=mean.device,dtype=mean.dtype,generator=g)
 if antithetic:noise=-noise
 y=mean+noise@L.T;tau=spec['smoothmax_tau'];smooth=tau*(torch.logsumexp(y/tau,-1)-math.log(FUT))/spec['scales'][1];cross=(y.max(-1).values>spec['spatial_mean_threshold']).to(mean.dtype)/spec['scales'][4]
 return torch.stack((analytic[0].expand(n),smooth,analytic[2].expand(n),analytic[3].expand(n),cross),-1)

@torch.no_grad()
def coarse_parameters_ssm(model,history):
 history=torch.as_tensor(history,device=model.U.device,dtype=model.U.dtype);m,P0,_=model.filter(history);A,Q,_,r=model.matrices();powers=[torch.eye(model.d,device=A.device,dtype=A.dtype)]
 for _ in range(FUT):powers.append(powers[-1]@A)
 transform=sum(powers[1:])/FUT;mean=m@transform.T;cov=transform@P0@transform.T
 for j in range(1,FUT+1):s=sum(powers[:FUT-j+1])/FUT;cov=cov+s@Q@s.T
 cov=(cov+cov.T)/2;return mean,torch.linalg.cholesky(cov),r

@torch.no_grad()
def draw_ssm_coarse(model,parameters,n,draw_seed,antithetic=False):
 mean,L,r=parameters;g=torch.Generator(device=mean.device).manual_seed(draw_seed);sign=-1 if antithetic else 1;pc=mean.repeat_interleave(n,0)+sign*torch.randn(len(mean)*n,model.d,device=mean.device,generator=g)@L.T;noise=sign*torch.randn(len(pc),P,device=mean.device,generator=g);value=(pc*model.scale)@model.U.T+(r/FUT).sqrt()*noise;return value.reshape(len(mean),n,1,P)

def allocate(variance,cost,budget):
 # Variances/costs are in level units (a pair for antithetic strategies).
 vf,va,v0,v1,va0,va1=variance;cf,cc,cp=cost;available=budget*cf
 choices=[{'policy':'mc','n0':0,'n1':budget,'predicted_variance':vf/budget,'predicted_sampling_seconds':budget*cf},{'policy':'antithetic_mc','n0':0,'n1':budget//2,'predicted_variance':va/(budget//2),'predicted_sampling_seconds':budget*cf}]
 if cc is not None:
  for name,w0,w1,c0,c1 in [('mlmc',v0,v1,cc,cp),('antithetic_mlmc',va0,va1,2*cc,2*cp)]:
   best=None
   for n1 in range(64,int(available/c1)//64*64+1,64):
    n0=int(max(0,available-n1*c1)/c0)//64*64
    if n0<64:continue
    value=w0/n0+w1/n1
    if best is None or value<best['predicted_variance']:best={'policy':name,'n0':n0,'n1':n1,'predicted_variance':value,'predicted_sampling_seconds':c0*n0+c1*n1}
   if best is not None:choices.append(best)
 return min(choices,key=lambda x:x['predicted_variance']),choices
