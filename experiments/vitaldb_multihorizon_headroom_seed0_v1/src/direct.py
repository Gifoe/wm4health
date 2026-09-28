"""Matched non-recursive direct predictors: the only difference is supervised horizons."""
import torch
from torch import nn
from common import H,F,CFG

class Direct(nn.Module):
    def __init__(self):
        super().__init__()
        d=64; a=CFG['direct_action_dim']; e=CFG['horizon_embedding_dim']
        # Exact history encoder and current-state residual design of the RSSM.
        self.encoder=nn.GRU(14,d,2,batch_first=True,dropout=.1)
        self.latent=nn.LayerNorm(d)
        # Causal prefix encoder: output at step h sees actions 1..h only.
        self.action_encoder=nn.GRU(4,a,batch_first=True)
        self.horizon=nn.Embedding(F,e)
        self.decoder=nn.Sequential(nn.Linear(d+a+e+4,64),nn.GELU(),nn.Linear(64,2))
        nn.init.zeros_(self.decoder[-1].weight);nn.init.zeros_(self.decoder[-1].bias)

    def forward(self,b,steps=None):
        x=torch.cat([b['state'],b['action'],b['demo'][:,None,:].expand(-1,H,-1)],-1)
        _,h=self.encoder(x);z=self.latent(h[-1])
        a,_=self.action_encoder(b['future_action'])
        if steps is None:steps=torch.arange(F,device=z.device)
        else:steps=torch.as_tensor(steps,device=z.device)-1
        embedding=self.horizon(steps)[None].expand(len(z),-1,-1)
        v=torch.cat([z[:,None].expand(-1,len(steps),-1),a[:,steps],embedding,
                     b['demo'][:,None].expand(-1,len(steps),-1)],-1)
        return b['current'][:,None]+self.decoder(v)
