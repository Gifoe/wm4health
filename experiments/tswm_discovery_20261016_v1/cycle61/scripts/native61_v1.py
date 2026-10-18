"""Unmodified NHSMM code, with a declared episode-definition support mask."""
from common61 import *
import sys
sys.path.insert(0,str(ROOT/'upstream/nhsmm'))
import torch
from nhsmm import NHSMM,ModelConfig,HSMMFilterRuntime
from nhsmm.filtering import filter_model_sequence,HSMMFilterState
from nhsmm.survival import active_episode_survival_forecast

def model(arm,seed,device='cpu',dtype=torch.float32,duration=32):
 assert arm in ['hsmm','hmm']
 cfg=ModelConfig(n_states=3,n_features=2,max_duration=duration if arm=='hsmm' else 1,causal=True,use_context_encoder=False,seed=seed,n_init=1,initial_init_mode='uniform',duration_init_mode='uniform',transition_init_mode='uniform',verbose=False)
 m=NHSMM(config=cfg,device=device).to(dtype=dtype);m.initialize_distributions()
 if arm=='hsmm':
  # An episode is a maximal constant-regime segment. Disallow invisible self-renewal.
  with torch.no_grad():m.dist.transition._structural_constraint.copy_((~torch.eye(3,device=device,dtype=torch.bool))[:,None,:].expand(3,duration,3))
 return m

def parameters(m):
 return (m.dist.initial.log_matrix(T=1)[0,0],m.dist.duration.log_matrix(T=1,soft_dmax=m.duration_logits_bias)[0,0],m.dist.transition.log_matrix(T=1,soft_dmax=m.duration_logits_bias)[0,0])
