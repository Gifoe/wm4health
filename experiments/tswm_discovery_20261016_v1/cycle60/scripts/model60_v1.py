"""Unchanged native neural-state components, with explicit post-update output timing."""
from common60 import *
import sys,torch
from typing import List
sys.path.insert(0,str(ROOT/'upstream/sysid'))
from torchid.ss.dt.models import NeuralLinStateUpdateV2,LinearOutput
from torchid.ss.dt.estimators import LSTMStateEstimator
from torchid.ss.dt.simulator import StateSpaceSimulator

class PostUpdateSSM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.f=NeuralLinStateUpdateV2(8,2,hidden_size=64)
        self.g=LinearOutput(8,1)
        self.estimator=LSTMStateEstimator(2,1,8,hidden_size=32)
    def forward(self,history_u:torch.Tensor,history_y:torch.Tensor,future_u:torch.Tensor):
        x=self.estimator(history_u,history_y)
        outputs=torch.jit.annotate(List[torch.Tensor],[])
        for k in range(future_u.size(0)):
            x=x+self.f(x,future_u[k])
            outputs.append(self.g(x))
        return torch.stack(outputs,0)
