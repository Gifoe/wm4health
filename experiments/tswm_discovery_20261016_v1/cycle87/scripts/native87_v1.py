from common87_v1 import *
M80=read(ROOT/'cycle80/outputs/SOURCE_MANIFEST80_V1.json');SOURCE=Path(M80['source_root']);DEPS=read(ROOT/'cycle80/outputs/DEPENDENCY_INSTALL80_V2.json')
sys.path.insert(0,DEPS['target']);sys.path.insert(0,str(SOURCE/'src'))
import numpy as np
import torch
torch.set_num_threads(2)
from fim.models.ode import load_fim_ode_local
ASSETS=read(ROOT/'cycle80/outputs/ASSET_MANIFEST80_V1.json')['assets'];WEIGHTS=next(a for a in ASSETS if a['relative'].endswith('.safetensors'));CONFIG=next(a for a in ASSETS if a['relative'].endswith('config.json'))
def check_native():
 for item in M80['files']:assert sha(SOURCE/item['path'])==item['sha256'],item['path']
 assert sha(WEIGHTS['path'])==WEIGHTS['sha256'];assert sha(CONFIG['path'])==CONFIG['sha256']
def load_native(device='cuda'):
 m=load_fim_ode_local(Path(WEIGHTS['path']).parent,device=device)
 for p in m.parameters():p.requires_grad_(True)
 assert sum(p.numel() for p in m.parameters())==12968068
 m.train_config.corruption_model_type=None
 # DataCorruptionModel captured the dispatch at construction; recreate native data preparation.
 from fim.models.ode_trainer import DataPreparation
 m.data_preparation=DataPreparation(m.train_config)
 return m
def pad(x):return np.pad(np.asarray(x,dtype=np.float32),[(0,0)]*(x.ndim-1)+[(0,1)])
def batch(context,locations,drift,drift_obs,device):
 b,t,n,_=context.shape;g=locations.shape[1]
 a=dict(obs_values=pad(context),obs_times=np.broadcast_to((np.arange(n)*.05)[None,None,:,None],(b,t,n,1)).copy().astype(np.float32),locations=pad(locations),obs_mask=np.ones((b,t,n,1),bool),dimension_mask=np.broadcast_to(np.array([True,True,False]),(b,g,3)).copy(),drift_at_locations=pad(drift),drift_at_observations=pad(drift_obs))
 return {k:torch.as_tensor(v,device=device) for k,v in a.items()}
def loss_value(out):
 # Native AModel stats wrapper returns loss in losses/loss.
 if 'loss' in out and torch.is_tensor(out['loss']):return out['loss']
 if 'losses' in out:
  x=out['losses'];return x['loss'] if isinstance(x,dict) else x
 raise KeyError(str(out.keys()))
def predict_field(model,context,locations,device='cuda'):
 b,t,n,_=context.shape;tt=np.broadcast_to((np.arange(n)*.05)[None,None,:,None],(b,t,n,1)).copy().astype(np.float32)
 with torch.no_grad():
  r=model.model_forward(torch.as_tensor(pad(context),device=device),torch.as_tensor(tt,device=device),torch.as_tensor(pad(locations),device=device),torch.ones((b,t,n,1),dtype=torch.bool,device=device))
  y=model.get_prediction_for_eval(r)[...,:2].cpu().numpy()
 return y
