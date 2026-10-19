"""Validate the ML quadrature implementation against an analytic Kalman model."""
import pathlib,sys,unittest
import numpy as np
import torch
from scipy.stats import norm
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'scripts'))
from generative_alarm_baseline import ObservationModel,grid_terms,loglik

def fixture():
 model=ObservationModel([4]).double()
 with torch.no_grad():
  for param in model.alarm.parameters():param.zero_()
  model.alarm[-1].bias.fill_(np.log(.2/.8));model.anchor_scale.fill_(1.2);model.anchor_bias.fill_(.1);model.log_noise.fill_(np.log(.4))
 obs=torch.tensor([[[.4,1,0],[-.2,1,1],[0.,0,0],[.8,1,0]],[[0.,0,1],[0.,0,1],[0.,0,0],[.1,1,0]]],dtype=torch.float64)
 return model,obs

class MLChecks(unittest.TestCase):
 def test_loglik_matches_analytic_kalman_with_missingness(self):
  model,obs=fixture();g,K,p=grid_terms(401,8.,.95,'cpu',torch.float64)
  numeric=loglik(model,obs,g,K,p).detach().numpy();exact=[]
  for seq in obs.numpy():
   mean=0.;variance=1.;score=0.
   for t,(x,mask,alarm) in enumerate(seq):
    if t:mean*=.95;variance=.95**2*variance+(1-.95**2)
    if mask:
     innovation=x-(1.2*mean+.1);S=1.2**2*variance+.4**2;score+=norm.logpdf(innovation,scale=np.sqrt(S));gain=variance*1.2/S;mean+=gain*innovation;variance-=gain*1.2*variance
    score+=np.log(.2 if alarm else .8)
   exact.append(score)
  np.testing.assert_allclose(numeric,exact,atol=1e-8,rtol=1e-8)
 def test_gradients_match_finite_differences(self):
  model,obs=fixture();g,K,p=grid_terms(129,6.,.95,'cpu',torch.float64)
  score=loglik(model,obs,g,K,p).sum();score.backward()
  for param in [model.anchor_scale,model.anchor_bias,model.log_noise,model.alarm[-1].bias]:
   autodiff=float(param.grad);original=param.detach().clone();step=1e-5
   with torch.no_grad():
    param.copy_(original+step);plus=float(loglik(model,obs,g,K,p).sum());param.copy_(original-step);minus=float(loglik(model,obs,g,K,p).sum());param.copy_(original)
   self.assertAlmostEqual(autodiff,(plus-minus)/(2*step),places=7)
if __name__=='__main__':unittest.main()
