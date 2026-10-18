"""Unmodified source with its supported injected-transition factory extension."""
from common61 import *
import sys
sys.path.insert(0,str(ROOT/'upstream/nhsmm'))
import torch
from nhsmm import NHSMM,ModelConfig,HSMMFilterRuntime,DistributionSet
from nhsmm.distributions import Transition
from nhsmm.filtering import filter_model_sequence,HSMMFilterState
from nhsmm.survival import active_episode_survival_forecast

class NoSelfTransition(Transition):
 def _build_structural_constraint(self):
  base=~torch.eye(self.n_states,dtype=torch.bool)
  return base if self.max_duration is None else base[:,None,:].expand(self.n_states,self.max_duration,self.n_states).clone()

class EpisodeDistributions(DistributionSet):
 def __init__(self,config):super().__init__(config=config,transition=NoSelfTransition)

def model(arm,seed,device='cpu',dtype=torch.float32,duration=32):
 assert arm in ['hsmm','hmm']
 cfg=ModelConfig(n_states=3,n_features=2,max_duration=duration if arm=='hsmm' else 1,causal=True,use_context_encoder=False,seed=seed,n_init=1,initial_init_mode='uniform',duration_init_mode='uniform',transition_init_mode='uniform',verbose=False)
 m=NHSMM(config=cfg,device=device).to(dtype=dtype)
 m.initialize_distributions(dist=EpisodeDistributions if arm=='hsmm' else None)
 return m

def parameters(m):
 return (m.dist.initial.log_matrix(T=1)[0,0],m.dist.duration.log_matrix(T=1,soft_dmax=m.duration_logits_bias)[0,0],m.dist.transition.log_matrix(T=1,soft_dmax=m.duration_logits_bias)[0,0])
