"""Same architectures, fixed-reference observation units. No per-context risk reweighting."""
import torch
from world_models import RecurrentWorld,NativeAttention,build
def scale_context(cy,cu,fu,target=None):
    return cy,cu,fu,target,torch.zeros_like(cy[:,:1]),torch.ones_like(cy[:,:1])
