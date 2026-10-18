import sys,pathlib
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common70_v1 import *
sys.path.insert(0,str(SOURCE))
import jax,jax.numpy as jnp,jax.random as jr,flax.linen as nn
from rp_ssm.actions import ContinuousActionMapper,DiscreteChoiceActionMapper,ActionNetwork
from rp_ssm.dists import ACLGParam,GaussianNatParam
from rp_ssm.distmaps import MVNDiag
from rp_ssm.datasets import TrainData
from rp_ssm.rpm import RPSSM,ConstrainedIVFreeEnergy

class DiscreteSignatureBridge:
 def __init__(self,mode,bias):self.mapper=DiscreteChoiceActionMapper(2);self.mode=mode;self.bias=bias
 def init(self,key,actions):return self.mapper.init(key,2,self.mode,self.bias,actions)
 def apply(self,actions,params):return self.mapper.apply(actions,params)

class ContinuousFlagBridge:
 def __init__(self,mapper,mode,bias):self.mapper=mapper;self.mode=mode;self.bias=bias
 def init(self,key,actions):return self.mapper.init(key,actions)
 def apply(self,actions,params):
  C,c=self.mapper.apply(actions,params)
  return jnp.zeros_like(C) if self.mode=='constant' else C,jnp.zeros_like(c) if self.bias=='none' else c

def preserving_update(prior,params):
 # Keep exactly the existing constructor and parameters; carry its dropped flags.
 return ACLGParam(action_mapper=prior.action_mapper,transition_matrix=prior.transition_matrix,transition_bias=prior.transition_bias,Q_dist_map=prior.Q_dist_map,opt_params=list(prior.opt_params),start_from_invariant=prior.start_from_invariant,**{**prior.params,**params})

H=np.array([[1.,.3],[-.2,.9]]);R=np.array([[.18,.02],[.02,.24]]);A=np.array([[.65,.12],[-.08,.76]]);B=np.array([.03,-.02]);M0=np.array([.4,-.7]);S0=np.array([[.9,.1],[.1,1.4]])
class FixedGaussianLikelihood:
 def apply(self,params,x):
  h=jnp.asarray(H);ri=jnp.asarray(np.linalg.inv(R));scale=x[2];return GaussianNatParam(p=scale*h.T@ri@h,pwm=scale*h.T@ri@x[:2])

def build(seed,kind,mode,invariant):
 rng=np.random.default_rng(seed);actions=rng.normal(0,.7,(4,31,2)) if kind=='continuous' else rng.integers(0,2,(4,31,1));actions=jnp.asarray(actions)
 if kind=='continuous':mapper=ContinuousActionMapper(ActionNetwork(network=nn.Sequential([nn.Dense(12),nn.tanh,nn.Dense(12)]),latent_dim=2))
 else:mapper=DiscreteSignatureBridge(mode,'action_dependent')
 prior=ACLGParam(action_mapper=mapper,transition_matrix=mode,transition_bias='action_dependent',Q_dist_map=MVNDiag(2),opt_params=['A','Q'],start_from_invariant=invariant,A=jnp.asarray(A),Q=jnp.array([-2.7,-3.0]),b=jnp.asarray(B),m1=jnp.asarray(M0),Q1=jnp.asarray(S0));params=prior.init(jr.PRNGKey(seed),actions)
 if kind=='discrete':
  Cs=np.array([[[-.05,.03],[.01,.02]],[[.07,-.04],[.02,-.03]]]);Cs=Cs+np.eye(2)[None] if mode=='multiplicative' else Cs;params['C']=jnp.asarray(Cs);params['c']=jnp.array([[-.2,.1],[.2,-.1]])
 native_updated=prior.update(params);fixed=preserving_update(prior,params);return actions,prior,params,native_updated,fixed

def joint_gaussian(m0,S0,As,bs,Q):
 T=len(As)+1;means=np.zeros((T,2));cov=np.zeros((T,2,T,2));means[0]=m0;cov[0,:,0,:]=S0
 for t in range(1,T):
  means[t]=As[t-1]@means[t-1]+bs[t-1]
  for s in range(t):cov[t,:,s,:]=As[t-1]@cov[t-1,:,s,:];cov[s,:,t,:]=cov[t,:,s,:].T
  cov[t,:,t,:]=As[t-1]@cov[t-1,:,t-1,:]@As[t-1].T+Q
 return means,cov.reshape(T*2,T*2)

def dense_condition(mean,cov,obs,scale):
 T=len(mean);p=np.zeros_like(cov);h=np.zeros(T*2);ri=np.linalg.inv(R)
 for t in range(T):p[2*t:2*t+2,2*t:2*t+2]=scale[t]*H.T@ri@H;h[2*t:2*t+2]=scale[t]*H.T@ri@obs[t]
 inv=np.linalg.inv(cov);post_cov=np.linalg.inv(inv+p);post_mean=post_cov@(inv@mean.ravel()+h);return post_mean.reshape(T,2),post_cov

def one_cell(seed,kind,mode,invariant):
 actions,prior,params,native_updated,fixed=build(seed,kind,mode,invariant);chain,inference=fixed.to_chain(32,actions);arrays={'actions':np.asarray(actions),'native_prior_mean':np.asarray(chain.params['means']),'native_prior_cov':np.asarray(chain.params['covs']),'native_prior_cross':np.asarray(chain.params['cross_covs']),'As':np.asarray(inference['As']),'bs':np.asarray(inference['bs']),'Q':np.asarray(fixed.params['Q']),'m0':np.asarray(fixed.params['m1']),'S0':np.asarray(fixed.params['Q1'])};rng=np.random.default_rng(seed+9);ys=[];scales=[];references=[];reference_cov=[];prior_refs=[];prior_covrefs=[]
 for i in range(4):
  m,c=joint_gaussian(arrays['m0'],arrays['S0'],arrays['As'][i],arrays['bs'][i],arrays['Q']);z=rng.multivariate_normal(m.ravel(),c).reshape(32,2);scale=np.where(rng.random(32)<.25,.02,1.);obs=np.stack([H@v+rng.multivariate_normal(np.zeros(2),R/s) for v,s in zip(z,scale)]);pm,pc=dense_condition(m,c,obs,scale);ys.append(obs);scales.append(scale);references.append(pm);reference_cov.append(pc);prior_refs.append(m);prior_covrefs.append(c)
 x=np.concatenate([np.array(ys),np.array(scales)[...,None]],-1);data=TrainData(obs=(jnp.asarray(x),),actions=actions);energy=ConstrainedIVFreeEnergy(RPSSM(fixed,[FixedGaussianLikelihood()]));_,factors,posterior=energy.get_posterior(None,fixed,[None],data);mm=chain.moment_match
 arrays.update(obs_y=np.asarray(ys),scale=np.asarray(scales),posterior_mean=np.asarray(posterior.params['means']),posterior_cov=np.asarray(posterior.params['covs']),posterior_cross=np.asarray(posterior.params['cross_covs']),dense_prior_mean=np.asarray(prior_refs),dense_prior_full_cov=np.asarray(prior_covrefs),dense_posterior_mean=np.asarray(references),dense_posterior_full_cov=np.asarray(reference_cov),moment_match_mean=np.asarray(mm.params['means']),moment_match_cov=np.asarray(mm.params['covs']),native_updated_m0=np.asarray(native_updated.params['m1']),native_updated_S0=np.asarray(native_updated.params['Q1']))
 # Exact supported Gaussian core, not the approximate empirical-RPM training objective.
 pd=np.array([[c[2*t:2*t+2,2*t:2*t+2] for t in range(32)] for c in prior_covrefs]);pc=np.array([[c[2*t:2*t+2,2*t:2*t+2] for t in range(32)] for c in reference_cov]);px=np.array([[c[2*(t+1):2*(t+2),2*t:2*t+2] for t in range(31)] for c in reference_cov]);prx=np.array([[c[2*(t+1):2*(t+2),2*t:2*t+2] for t in range(31)] for c in prior_covrefs]);delta=np.asarray(prior_refs)-np.mean(prior_refs,0);match_cov=np.mean(pd+delta[:,:,:,None]*delta[:,:,None,:],0)
 errors=dict(prior_mean=float(abs(arrays['native_prior_mean']-prior_refs).max()),prior_cov=float(abs(arrays['native_prior_cov']-pd).max()),prior_cross=float(abs(arrays['native_prior_cross']-prx).max()),posterior_mean=float(abs(arrays['posterior_mean']-references).max()),posterior_cov_minus_padding=float(abs(arrays['posterior_cov']-np.eye(2)*1e-5-pc).max()),posterior_cross_raw=float(abs(arrays['posterior_cross']-px).max()),moment_match_mean=float(abs(arrays['moment_match_mean']-np.mean(prior_refs,0)).max()),moment_match_cov=float(abs(arrays['moment_match_cov']-match_cov).max()))
 for name,err in errors.items():assert err<=(1e-4 if name=='posterior_cross_raw' else 1e-7),(name,err)
 # Prefix-only inference then two conditional future plans, with unchanged known system parameters.
 prefix=TrainData(obs=(jnp.asarray(x[:,:16]),),actions=actions[:,:15]);_,_,post=energy.get_posterior(None,fixed,[None],prefix);fm=[];fc=[];fref=[];fcref=[]
 for plan in [np.asarray(actions)[:,15:],-np.asarray(actions)[:,15:] if kind=='continuous' else 1-np.asarray(actions)[:,15:]]:
  plan_m=[];plan_c=[];ref_m=[];ref_c=[]
  for i in range(4):
   m=np.asarray(post.params['means'][i,-1]);cov=np.asarray(post.params['covs'][i,-1])-1e-5*np.eye(2);fresh=ACLGParam(action_mapper=fixed.action_mapper,transition_matrix=mode,transition_bias='action_dependent',Q_dist_map=None,opt_params=['A'],start_from_invariant=False,**{**fixed.params,'m1':jnp.asarray(m),'Q1':jnp.asarray(cov)});ch,ip=fresh.to_chain(17,jnp.asarray(plan[i:i+1]));rr,cc=joint_gaussian(m,cov,np.asarray(ip['As'][0]),np.asarray(ip['bs'][0]),arrays['Q']);diag=np.array([cc[2*t:2*t+2,2*t:2*t+2] for t in range(17)]);plan_m.append(np.asarray(ch.params['means'][0]));plan_c.append(np.asarray(ch.params['covs'][0]));ref_m.append(rr);ref_c.append(diag)
  fm.append(plan_m);fc.append(plan_c);fref.append(ref_m);fcref.append(ref_c)
 arrays.update(forecast_mean=np.asarray(fm),forecast_cov=np.asarray(fc),forecast_reference_mean=np.asarray(fref),forecast_reference_cov=np.asarray(fcref));errors['forecast_mean']=float(abs(arrays['forecast_mean']-arrays['forecast_reference_mean']).max());errors['forecast_cov']=float(abs(arrays['forecast_cov']-arrays['forecast_reference_cov']).max());assert errors['forecast_mean']<1e-7 and errors['forecast_cov']<1e-7
 return arrays,dict(seed=seed,kind=kind,mode=mode,start_from_invariant=invariant,native_update_retained_flag=native_updated.start_from_invariant==invariant,native_updated_initial_mean_error=float(abs(arrays['native_updated_m0']-arrays['m0']).max()),native_updated_initial_cov_error=float(abs(arrays['native_updated_S0']-arrays['S0']).max()),errors=errors,component_only_no_learning=True,ordinary_adapters=['discrete signature bridge'] if kind=='discrete' else [],state_flag_adapter_used=not invariant)
