"""Check normalized joint likelihood and gradients independently of training."""
import itertools,pathlib,sys,unittest
import numpy as np
import torch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'scripts'))
from rpm_exact_baseline import Recognition,ratio_loglik
from generative_alarm_baseline import grid_terms

class RPMChecks(unittest.TestCase):
 def test_joint_normalizes_over_all_observation_sequences(self):
  torch.manual_seed(3);g,K,p=grid_terms(401,8.,.95,'cpu',torch.float64)
  reference=torch.tensor([[[1.,1.,0.],[-1.,1.,1.]],[[0.,0.,1.],[.5,1.,0.]],[[-.5,1.,0.],[0.,0.,1.]]],dtype=torch.float64)
  query=torch.stack([torch.stack([reference[i,0],reference[j,1]]) for i,j in itertools.product(range(3),repeat=2)])
  for kind in ['positive_site','full_gaussian']:
   model=Recognition(kind,[4]).double();scores=ratio_loglik(model,query,g,K,p,reference)
   self.assertAlmostEqual(float(scores.exp().sum()/9),1.,places=9)
 def test_gradient_of_reference_normalizer_is_included(self):
  torch.manual_seed(4);g,K,p=grid_terms(129,6.,.95,'cpu',torch.float64);x=torch.randn(4,3,3,dtype=torch.float64)
  for kind in ['positive_site','full_gaussian']:
   model=Recognition(kind,[4]).double();loss=ratio_loglik(model,x,g,K,p).mean();loss.backward();param=model.network[-1].bias
   for index in [0,1]:
    expected=float(param.grad[index]);original=float(param[index].detach());step=1e-5
    with torch.no_grad():
     param[index]=original+step;plus=float(ratio_loglik(model,x,g,K,p).mean());param[index]=original-step;minus=float(ratio_loglik(model,x,g,K,p).mean());param[index]=original
    self.assertAlmostEqual(expected,(plus-minus)/(2*step),places=7)
if __name__=='__main__':unittest.main()
