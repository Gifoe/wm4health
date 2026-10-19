"""Known context conditioning / low-rank hypernetwork baselines; no novelty claim."""
import pathlib,sys,math
import torch
from torch import nn
from torch.nn import functional as F
ROOT=pathlib.Path(__file__).resolve().parents[1]

class RecurrentWorld(nn.Module):
    def __init__(self,kind):
        super().__init__();self.kind=kind;self.width={'compiled64':64,'conditional96':96,'conditional128':128}[kind]
        self.encoder=nn.GRU(2,96,batch_first=True);self.code=nn.Linear(96,64);self.init=nn.Linear(96,self.width)
        self.cell=nn.GRUCell(1 if kind=='compiled64' else 65,self.width);self.readout=nn.Linear(self.width,1)
        if kind=='compiled64':
            self.rank=4;self.left=nn.Linear(64,3*self.width*self.rank);self.right=nn.Linear(64,self.rank*self.width)
            nn.init.normal_(self.left.weight,std=.01);nn.init.zeros_(self.left.bias);nn.init.normal_(self.right.weight,std=.01);nn.init.zeros_(self.right.bias)
    def encode(self,cy,cu):
        _,state=self.encoder(torch.cat((cy,cu),-1));e=state[-1];code=torch.tanh(self.code(e));h=torch.tanh(self.init(e))
        if self.kind=='compiled64':
            delta=torch.bmm(self.left(code).reshape(-1,3*self.width,self.rank),self.right(code).reshape(-1,self.rank,self.width))/math.sqrt(self.rank)
            return h,delta
        return h,code
    def advance(self,h,u,conditioning):
        if self.kind!='compiled64':return self.cell(torch.cat((u,conditioning),-1),h)
        gi=F.linear(u,self.cell.weight_ih,self.cell.bias_ih)
        gh=F.linear(h,self.cell.weight_hh,self.cell.bias_hh)+torch.bmm(conditioning,h.unsqueeze(-1)).squeeze(-1)
        ir,iz,inn=gi.chunk(3,-1);hr,hz,hn=gh.chunk(3,-1)
        reset=torch.sigmoid(ir+hr);update=torch.sigmoid(iz+hz);candidate=torch.tanh(inn+reset*hn)
        return candidate+update*(h-candidate)
    def simulate(self,encoded,fu):
        h,condition=encoded;pred=[]
        for t in range(fu.shape[1]):
            h=self.advance(h,fu[:,t],condition);pred.append(self.readout(h))
        return torch.stack(pred,1)
    def forward(self,cy,cu,fu):return self.simulate(self.encode(cy,cu),fu)

class NativeAttention(nn.Module):
    def __init__(self):
        super().__init__();sys.path.insert(0,str(ROOT.parent/'cycle16/upstream/sysid_benchmarks'))
        from transformer_sim import Config,TSTransformer
        self.config=dict(seq_len_ctx=128,seq_len_new=48,seq_len_patch=128,d_model_RNN=32,n_layer=2,n_head=4,n_embd=64,n_u=1,n_y=1,dropout=0.,bias=False,device_name='cuda')
        self.core=TSTransformer(Config(**self.config));self.n_in=16
        for name,p in self.core.named_parameters():
            if name.startswith(('before_enc_wte.','RNN.','encoder_wte.','encoder_wte2.','lm_head_logvar.')):p.requires_grad_(False)
    def encode(self,cy,cu):return self.core.encoder(self.core.embed_ctx(cy,cu)),cy[:,-self.n_in:],cu[:,-self.n_in:]
    def simulate(self,encoded,fu):
        mem,initial_y,initial_u=encoded
        u=torch.cat((initial_u,fu),1);y=torch.cat((initial_y,torch.zeros_like(fu)),1)
        query=self.core.embed_new(u,y,self.n_in);out=self.core.decoder(query,mem)[:,self.n_in:]
        return self.core.lm_head_mean(out)
    def forward(self,cy,cu,fu):return self.simulate(self.encode(cy,cu),fu)

def build(kind):return NativeAttention() if kind=='native_attention64' else RecurrentWorld(kind)
def scale_context(cy,cu,fu,target=None):
    ym=cy.mean(1,keepdim=True);ys=cy.std(1,keepdim=True,unbiased=False).clamp_min(1e-3)
    um=cu.mean(1,keepdim=True);us=cu.std(1,keepdim=True,unbiased=False).clamp_min(1e-3)
    return (cy-ym)/ys,(cu-um)/us,(fu-um)/us,None if target is None else (target-ym)/ys,ym,ys
