"""Ordinary parameter-comparable GRU control; no new mechanism."""
from common60 import *
import torch
class GRUControl(torch.nn.Module):
    def __init__(self):
        super().__init__();self.encoder=torch.nn.GRU(3,30);self.decoder=torch.nn.GRU(2,30);self.output=torch.nn.Linear(30,1)
    def forward(self,history_u,history_y,future_u):
        _,hidden=self.encoder(torch.cat([history_u,history_y],dim=-1))
        states,_=self.decoder(future_u,hidden)
        return self.output(states)
