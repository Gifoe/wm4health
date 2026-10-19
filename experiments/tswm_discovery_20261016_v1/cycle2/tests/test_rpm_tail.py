"""Independent checks of the probability model, variational cap and quadrature."""
import pathlib,sys,unittest
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.special import logsumexp
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'scripts'))
from probe_rpm_tail_bound import log_normal,log_unnormalized,solve

class TailChecks(unittest.TestCase):
 def test_observation_likelihoods_normalize_pointwise(self):
  z=np.linspace(-30,30,2001);V=9.;eps=.001
  a=np.log1p(-eps)+log_normal(z,1.);b=np.log(eps)+log_normal(z,V);F=np.logaddexp(a,b)
  np.testing.assert_allclose(np.exp(a-F)+np.exp(b-F),1.,atol=3e-14)
 def test_joint_marginal_evidence_normalizes(self):
  eps=.01;V=3.
  rare=quad(lambda z:eps*np.exp(log_unnormalized(z,V,eps)),-np.inf,np.inf,epsabs=1e-11)[0]
  common=quad(lambda z:np.exp(log_normal(z,1.))*(1-eps)*np.exp(log_normal(z,1.)-np.logaddexp(np.log1p(-eps)+log_normal(z,1.),np.log(eps)+log_normal(z,V))),-np.inf,np.inf,epsabs=1e-11)[0]
  self.assertAlmostEqual(rare+common,1.,places=10)
 def test_full_auxiliary_optimization_matches_centered_formula(self):
  V=3.;eps=.001;row=solve(V,eps,[160,320])
  def objective(x):
   h=-1/V+np.exp(x[0]);linear=x[1];P=1+1/V+h
   logGamma=logsumexp([np.log1p(-eps)-.5*np.log1p(h)+linear**2/(2*(1+h)),np.log(eps)-.5*np.log1p(V*h)+linear**2/(2*(1/V+h))])
   return -(np.log(2)-.5*np.log(V)-np.log(P)+linear**2/(2*P)-logGamma)
  for linear in [-2.,-.2,.2,2.]:
   res=minimize(objective,[-4.,linear],method='BFGS',options={'gtol':1e-8})
   self.assertLess(abs(res.x[1]),2e-5)
   self.assertAlmostEqual(row['interior_objective_gap'],row['exact_log_evidence_without_log_reference']+res.fun,places=7)
 def test_rare_limit_separates_gaussian_family_from_auxiliary_gap(self):
  row=solve(3.,1e-10,[320,640])
  self.assertGreater(row['exact_variance'],2.999)
  self.assertLess(row['ordinary_gaussian_vi'][-1]['kl_to_exact'],1e-4)
  self.assertLess(row['interior_q_variance'],2.)
  self.assertGreater(row['interior_objective_gap'],.058)
 def test_no_gap_control_below_capacity_ceiling(self):
  row=solve(1.5,1e-5,[160,320])
  self.assertLess(row['interior_objective_gap'],1e-8)
  self.assertLess(abs(row['ordinary_gaussian_vi'][-1]['variance']-row['interior_q_variance']),1e-5)
if __name__=='__main__':unittest.main()
