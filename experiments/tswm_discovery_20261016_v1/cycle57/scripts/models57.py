"""Matched recognition/prior model and a structured proper-likelihood control."""
import torch
from torch import nn
from math57 import raw_rho,rho_from

class Neural(nn.Module):
    def __init__(self):
        super().__init__()
        self.theta=nn.Parameter(torch.tensor([raw_rho(.5),0.,0.]))
        self.log_lam=nn.Parameter(torch.zeros(2))
        self.network=nn.Sequential(nn.Linear(2,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,2))
        with torch.no_grad():
            self.network[-1].weight.mul_(.1);self.network[-1].bias.zero_()
    def encode(self,x):return self.network(x)
    @property
    def lam(self):return self.log_lam.exp()

class Proper(nn.Module):
    def __init__(self):
        super().__init__()
        self.theta=nn.Parameter(torch.tensor([raw_rho(.5),0.,0.]))
        self.log_lam=nn.Parameter(torch.zeros(2))
        self.raw_k=nn.Parameter(torch.full((2,),-2.1972245773362196))
    @property
    def lam(self):return self.log_lam.exp()
    @property
    def k(self):return 2*self.raw_k.sigmoid()
    def encode(self,x):return torch.sinh(self.k*x)/self.k

def describe(model):
    return {'rho':float(rho_from(model.theta[0]).detach()),'angles':model.theta[1:3].detach().cpu().tolist(),'site_precision':model.lam.detach().cpu().tolist(),'parameters':sum(p.numel() for p in model.parameters())}
