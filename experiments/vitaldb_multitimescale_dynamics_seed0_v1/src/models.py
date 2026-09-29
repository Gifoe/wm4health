"""Direct-MH capacity control and explicit fast/slow 10s/60s predictor."""
import sys
from pathlib import Path
import numpy as np,torch,yaml
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
CFG=yaml.safe_load((ROOT/'config.yaml').read_text())
SOURCE=(ROOT/CFG['source_experiment']).resolve()
DIRECT=(ROOT/CFG['direct_experiment']).resolve()
sys.path.insert(0,str(SOURCE/'src'))
from core import Store,Windows,to_device,seed_all,H,F,RSSM
sys.path.insert(0,str(DIRECT/'src'))
from direct import Direct

STEPS=CFG['horizons']

class DirectCapacity(nn.Module):
    """Same single-state Direct-MH formulation with wider history GRU only."""
    def __init__(self,d=73):
        super().__init__();self.d=d
        self.encoder=nn.GRU(14,d,2,batch_first=True,dropout=.1)
        self.latent=nn.LayerNorm(d)
        self.action_encoder=nn.GRU(4,32,batch_first=True)
        self.horizon=nn.Embedding(F,8)
        self.decoder=nn.Sequential(nn.Linear(d+32+8+4,64),nn.GELU(),nn.Linear(64,2))
        nn.init.zeros_(self.decoder[-1].weight);nn.init.zeros_(self.decoder[-1].bias)
    def encode(self,b):
        x=torch.cat([b['state'],b['action'],b['demo'][:,None,:].expand(-1,H,-1)],-1)
        _,h=self.encoder(x);return self.latent(h[-1])
    def forward(self,b,steps=STEPS,return_latents=False):
        z=self.encode(b);a,_=self.action_encoder(b['future_action'])
        ix=torch.as_tensor(steps,device=z.device)-1
        q=torch.cat([z[:,None].expand(-1,len(ix),-1),a[:,ix],self.horizon(ix)[None].expand(len(z),-1,-1),
                     b['demo'][:,None].expand(-1,len(ix),-1)],-1)
        pred=b['current'][:,None]+self.decoder(q)
        return (pred,{'shared':z}) if return_latents else pred

class MTDynamics(nn.Module):
    """Causal fast GRUCell each 10s; slow GRUCell only on complete 60s blocks."""
    def __init__(self):
        super().__init__();d=64;f=32;s=32
        self.encoder=nn.GRU(14,d,2,batch_first=True,dropout=.1)
        self.latent=nn.LayerNorm(d)
        self.fast_init=nn.Linear(d,f)
        self.slow_init=nn.Linear(d,s)
        self.fast_transition=nn.GRUCell(4+s+4,f)
        self.slow_transition=nn.GRUCell(4+f+4,s)
        self.horizon=nn.Embedding(F,8)
        self.decoder=nn.Sequential(nn.Linear(f+s+8+4,64),nn.GELU(),nn.Linear(64,2))
        nn.init.zeros_(self.decoder[-1].weight);nn.init.zeros_(self.decoder[-1].bias)
    def encode(self,b):
        x=torch.cat([b['state'],b['action'],b['demo'][:,None,:].expand(-1,H,-1)],-1)
        _,h=self.encoder(x);z=self.latent(h[-1])
        return self.fast_init(z),self.slow_init(z)
    def forward(self,b,steps=STEPS,return_latents=False,swap_fast=None,swap_slow=None):
        fast,slow=self.encode(b)
        if swap_fast is not None:fast=swap_fast
        if swap_slow is not None:slow=swap_slow
        initial=(fast,slow);actions=b['future_action'];ys=[];snapshots={}
        wanted=set(int(x) for x in steps)
        for t in range(1,F+1):
            fast=self.fast_transition(torch.cat([actions[:,t-1],slow,b['demo']],-1),fast)
            if t%CFG['slow_stride_steps']==0:
                block=actions[:,t-CFG['slow_stride_steps']:t,:].mean(1)
                slow=self.slow_transition(torch.cat([block,fast,b['demo']],-1),slow)
            if t in wanted:
                emb=self.horizon(torch.tensor(t-1,device=fast.device)).expand(len(fast),-1)
                decoded=self.decoder(torch.cat([fast,slow,emb,b['demo']],-1))
                snapshots[t]=b['current']+decoded
        pred=torch.stack([snapshots[int(h)] for h in steps],1)
        return (pred,{'fast':initial[0],'slow':initial[1]}) if return_latents else pred

def parameters(model):return sum(p.numel() for p in model.parameters())
