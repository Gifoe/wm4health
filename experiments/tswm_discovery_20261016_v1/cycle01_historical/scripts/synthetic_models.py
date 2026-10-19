"""Official WorldTS core and plainly labelled known diagnostic baselines."""
import importlib.util,pathlib,sys
from types import SimpleNamespace
import torch
from torch import nn
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'upstream/WorldTS/DAG-bench'))
from ts_benchmark.baselines.worldts.models.worldts_model import WorldTSModel
from ts_benchmark.baselines.worldts.losses.latent_state_loss import LatentStateLoss

class FIRCondition(nn.Module):
    """Ordinary causal convolution; explicitly NOT a proposed new method."""
    def __init__(self,local,width,horizon):
        super().__init__();self.local=local;self.fir=nn.Conv1d(1,width,horizon,bias=False)
        nn.init.zeros_(self.fir.weight);self.horizon=horizon
    def forward(self,u):
        v=nn.functional.pad(u.transpose(1,2),(self.horizon-1,0))
        return self.local(u)+self.fir(v).transpose(1,2)

class Native(nn.Module):
    def __init__(self,kind,L,H,d=16):
        super().__init__();self.kind=kind
        cfg=SimpleNamespace(seq_len=L,pred_len=H,horizon=H,input_dim=2,output_dim=1,series_dim=1,
            future_x_dim=1,latent_dim=d,condition_dim=d,state_patch_len=3,state_hidden_multiplier=2,
            decoder_type='causal_patch',decoder_patch_len=3,decoder_hidden_multiplier=2,
            predictor_patch_len=8,predictor_patch_stride=4,predictor_d_model=32,predictor_n_heads=4,
            predictor_e_layers=1,predictor_d_ff=64,predictor_dropout=0.,use_revin=False,
            use_future_x_condition=True,use_time_condition=False,use_modal_condition=False)
        self.core=WorldTSModel(cfg)
        if kind=='worldts_fir':self.core.condition_encoder=FIRCondition(self.core.condition_encoder,d,H)
    def forward(self,h,u):return self.core.forecast(h,x_future=u)

class GRUModel(nn.Module):
    """Standard recurrent controlled state model, not claimed novel."""
    def __init__(self,d=48):
        super().__init__();self.history=nn.GRU(2,d,batch_first=True)
        self.transition=nn.GRU(1,d,batch_first=True);self.out=nn.Linear(d,1)
    def forward(self,h,u):
        _,z=self.history(h);states,_=self.transition(u,z);return self.out(states)

def make_model(name,cfg):
    if name=='gru':return GRUModel()
    return Native(name,cfg['history'],cfg['horizon'],cfg['latent_dim'])

def stage1_loss(out):
    loss=LatentStateLoss(latent_mse_weight=10.,latent_alignment_loss='mae',
        latent_cosine_weight=15.,latent_sigreg_weight=.1,latent_detach_target=True)
    return loss(h_pred=out['h_pred'],h_target=out['h_target'])['loss']
