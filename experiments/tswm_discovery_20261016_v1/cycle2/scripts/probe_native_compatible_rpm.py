"""A fixed RPM within the shipped positive-Gaussian-site parameterization.

This is a possibility/capacity audit, never reported as a trained model result.
Full recognition variances are <= the stationary prior variance. Unlike the
small-TV construction, its reference mixture is not arbitrarily close to prior.
"""
from probe_native_rp_precision import ROOT,FixedRecognition
import hashlib,json,pathlib,time
import numpy as onp
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp,roots_hermitenorm
from scipy.stats import norm
import jax.numpy as np
from rp_ssm.dists import LGStationaryParam
from rp_ssm.datasets import TrainData
from rp_ssm.rpm import RPSSM,ConstrainedIVFreeEnergy

def solve(c,r,eps,native_variance):
 def ln(z,v):return -.5*(onp.log(2*onp.pi*v)+z*z/v)
 def logu(z):return ln(z,1)+ln(z,r)-onp.logaddexp(onp.log1p(-eps)+ln(z,c),onp.log(eps)+ln(z,r))
 density=lambda z:onp.exp(logu(z))
 Z,zerr=quad(density,-onp.inf,onp.inf,epsabs=1e-11,epsrel=1e-11,limit=300)
 v=quad(lambda z:z*z*density(z),-onp.inf,onp.inf,epsabs=1e-10,epsrel=1e-10,limit=300)[0]/Z
 nodes,weights=roots_hermitenorm(320);weights/=onp.sqrt(2*onp.pi)
 def elbo(logv):
  vv=onp.exp(logv)
  return weights@logu(nodes*onp.sqrt(vv))+.5*onp.log(2*onp.pi*onp.e*vv)
 opt=minimize_scalar(lambda lv:-elbo(lv),bounds=(-8,onp.log(20)),method='bounded')
 def iv(t):
  h=-1/r+onp.exp(t);P=1+1/r+h
  Gamma=logsumexp([onp.log1p(-eps)-.5*onp.log1p(c*h),onp.log(eps)-.5*onp.log1p(r*h)])
  G=onp.log(2)-.5*onp.log(r)-onp.log(P)-Gamma
  return G,2/P
 ivopt=minimize_scalar(lambda t:-iv(t)[0],bounds=(-27,10),method='bounded');G,viv=iv(ivopt.x)
 future=[]
 for H in [0,1,8,32]:
  a=.95**H;q=1-a*a;nv=a*a*native_variance+q;ev=a*a*v+q
  if H==0:
   coverage=quad(density,-1.959963984540054*onp.sqrt(nv),1.959963984540054*onp.sqrt(nv),epsabs=1e-10)[0]/Z
   tail=1-quad(density,-2,2,epsabs=1e-10)[0]/Z
  else:
   sd=onp.sqrt(q);radius=1.959963984540054*onp.sqrt(nv)
   coverage=quad(lambda z:density(z)*(norm.cdf((radius-a*z)/sd)-norm.cdf((-radius-a*z)/sd)),-onp.inf,onp.inf,epsabs=1e-9)[0]/Z
   tail=quad(lambda z:density(z)*(norm.cdf((-2-a*z)/sd)+norm.sf((2-a*z)/sd)),-onp.inf,onp.inf,epsabs=1e-9)[0]/Z
  future.append({'horizon':H,'exact_forecast_variance':float(ev),'native_forecast_variance':float(nv),'true_coverage_of_native_gaussian_95_interval':float(coverage),'exact_tail_probability_abs_gt2':float(tail),'native_tail_probability_abs_gt2':float(2*norm.sf(2/onp.sqrt(nv)))})
 return {'common_full_recognition_variance':c,'rare_full_recognition_variance':r,'reference_rare_probability':eps,'implied_rare_probability':float(eps*Z),
  'exact_posterior_variance':float(v),'native_posterior_variance':float(native_variance),'ordinary_gaussian_vi_variance':float(onp.exp(opt.x)),
  'ordinary_gaussian_vi_kl':float(onp.log(Z)+opt.fun),'general_gaussian_interior_variance':float(viv),'general_gaussian_interior_objective_gap':float(onp.log(Z)-G),
  'exact_log_concavity_margin':float(1-(1/c-1/r)),'future':future,'evidence_quad_error':float(zerr)}

def main():
 start=time.perf_counter();prior=LGStationaryParam(start_from_invariant=True,stay_at_invariant=True,opt_params=['A'],A=np.array([[.95]]));energy=ConstrainedIVFreeEnergy(RPSSM(prior,[FixedRecognition()]))
 records=[]
 for c,r in [(.6,.999),(.75,.999),(.6,.6)]:
  precisions=onp.array([1/c-1,1/r-1]);logits=onp.log(onp.expm1(precisions))
  x=onp.zeros((2,8,2));x[:,:,1]=-30.;x[:,-1,1]=logits
  post=energy.get_posterior(None,prior,[None],TrainData(obs=(np.array(x),)))[2].params
  native_variance=float(onp.asarray(post['covs'])[1,-1,0,0]-1e-5)
  for eps in [.1,.001,1e-5]:
   row=solve(c,r,eps,native_variance);records.append(row)
   assert row['exact_log_concavity_margin']>0
   assert row['general_gaussian_interior_objective_gap']>=row['ordinary_gaussian_vi_kl']-1e-7
   print(json.dumps({k:row[k] for k in ['common_full_recognition_variance','rare_full_recognition_variance','reference_rare_probability','exact_posterior_variance','native_posterior_variance','ordinary_gaussian_vi_kl']}),flush=True)
 result={'scope':'New fixed oracle-recognition component audit in the native model class. Does not show learned parameters choose this model, and does not inherit the separate small-TV claim.','records':records,'code_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'runtime_seconds':time.perf_counter()-start}
 (ROOT/'outputs/native_compatible_rpm_probe.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
if __name__=='__main__':main()
