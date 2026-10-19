"""Author SPF computational bodies in a minimal, explicit runtime context."""
import sys,pathlib,os,ast,types,json,hashlib,math,typing
R=pathlib.Path(__file__).resolve().parents[1];B=R.parent
sys.path.insert(0,str(B/'cycle29/scripts'));import runtime_profile
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',MPLBACKEND='Agg')
sys.path.insert(0,str(R/'vendor'));S=R/'source_archive/spf';sys.path.insert(0,str(S))
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from diffusers.utils.torch_utils import randn_tensor
from emulator.models.spatiotemporal_pyramid_flow.model import SpatiotemporalPyramidFlow as Backbone
from emulator.schedulers.temporal_flow import TemporalFlowMatchEulerDiscreteScheduler as Scheduler
from emulator.data.loading.collate import collate_fn_spf
torch.set_num_threads(1)
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def seed(s):torch.manual_seed(s);torch.cuda.manual_seed_all(s);np.random.seed(s)
def dump(p,o):pathlib.Path(p).write_text(json.dumps(o,indent=2),encoding='utf-8')
class Attr(dict):
 __getattr__=dict.__getitem__
def attr(**kwargs):return Attr(kwargs)
SOURCE=S/'emulator/models/spatiotemporal_pyramid_flow_model.py'
def methods(repair=False):
 tree=ast.parse(SOURCE.read_text(encoding='utf-8'));wanted={'training_step','sample_block_noise','predict','sample_block_noise_seeded','predict_modular'}
 distribution=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='MultivariateNormalWithGenerator')
 original=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='SpatiotemporalPyramidFlow')
 bodies=[x for x in original.body if isinstance(x,ast.FunctionDef) and x.name in wanted]
 hashes={x.name:hashlib.sha256(ast.dump(x,include_attributes=False).encode()).hexdigest() for x in bodies}
 if repair:
  # Only initialize optional returned metadata. Does not change generated samples.
  fn=next(x for x in bodies if x.name=='predict_modular');fn.body.insert(0,ast.parse('indices = None').body[0])
 cls=ast.ClassDef(name='NativeMethods',bases=[],keywords=[],body=bodies,decorator_list=[])
 module=ast.fix_missing_locations(ast.Module(body=[distribution,cls],type_ignores=[]))
 namespace={'math':math,'torch':torch,'F':F,'rearrange':rearrange,'randn_tensor':randn_tensor,'Any':typing.Any,'List':typing.List,'Optional':typing.Optional,'MultivariateNormal':torch.distributions.MultivariateNormal,'_batch_mv':torch.distributions.multivariate_normal._batch_mv}
 exec(compile(module,str(SOURCE),'exec'),namespace)
 return namespace['NativeMethods'],hashes
def make_context(device='cuda',repair=False):
 cls,hashes=methods(repair);ctx=cls();ctx.device=torch.device(device)
 ctx.model=Backbone(height=8,width=8,cond_height=8,cond_width=8,in_channels=1,cond_in_channels=1,out_channels=1,num_attention_heads=2,attention_head_dim=32,depth=1,patch_size=2,cond_patch_size=2,max_num_frames=32,cross_attention_dim=64,pos_embed_type='sincos',temp_pos_embed_type='sincos',temp_cond_pos_embed_type='sincos',use_grad_checkpointing=False,add_temp_pos_embed=True,use_flash_attn=False).to(device)
 ctx.noise_scheduler=Scheduler(ns=[[4,4],[4,4],[4,64]],num_stages=3,stage_range=[0,.33,.66,1],num_train_timesteps=1000,shift=1.)
 ctx.noise_scheduler_config=attr(_target_='emulator.schedulers.temporal_flow.TemporalFlowMatchEulerDiscreteScheduler',num_stages=3,ns=[[4,4],[4,4],[4,64]])
 ctx.datamodule_config=attr(output_variables=['x'],timescales=['decadal','yearly','monthly'],output_seq_len={'decadal':1,'yearly':1,'monthly':16})
 ctx.stages=['decadal','yearly','monthly'];ctx.climate_models=['synthetic'];ctx.num_inference_steps=2;ctx.multi_scale=True;ctx.config={'model':{'static_condition':False,'use_pathway':False,'no_time_embedding':False}}
 grid=attr(lon=np.arange(8),lat=np.arange(8));ctx.trainer=attr(validating=False,datamodule=attr(test_ds=attr(climate_model_datasets=[attr(grid=grid)])))
 ctx.n_params=sum(p.numel() for p in ctx.model.parameters());ctx.parameters=ctx.model.parameters;ctx.log_dict=lambda *args,**kwargs:None
 return ctx,hashes
def toy_examples(n=12):
 # Initial observed field -> stable noisy nonlinear evolution. No physical claims.
 rng=np.random.default_rng(202643);examples=[]
 for j in range(n):
  first=rng.normal(size=(1,8,8)).astype('float32');state=first.copy();future=[]
  for t in range(16):state=.85*state+.1*np.tanh(state)+rng.normal(0,.15,size=state.shape);future.append(state.copy())
  future=np.asarray(future,dtype='float32');mean=future.mean(0,keepdims=True);ex={'metadata':{'climate_model':'synthetic'}}
  for key,y in [('decadal',mean),('yearly',mean),('monthly',future)]:
   x=np.repeat(first[None],len(y),axis=0)
   ex[key]={'X':torch.tensor(x),'Y':torch.tensor(y),'metadata':{'input_stats':[0.,1.],'stats':[0.,1.],'X_time':[0,0],'Y_time':[0,0]}}
  examples.append(ex)
 return examples
def collate_config():return attr(model=attr(params=attr(patch_size=2,same_scale_prob=.5,no_decadal=False,use_pathway=False,multiscale_warmup=0,multiscale_ramp=1)),scheduler=attr(num_stages=3,uses_temporal_levels=True))
def to_device(value,device):
 if torch.is_tensor(value):return value.to(device)
 if isinstance(value,dict):return {k:to_device(v,device) for k,v in value.items()}
 if isinstance(value,list):return [to_device(v,device) for v in value]
 return value
def prediction_inputs(examples,device):
 xs={k:torch.stack([x[k]['X'] for x in examples]).to(device) for k in ['decadal','yearly','monthly']}
 metadata={k+'_Y_time':[torch.zeros(len(examples),dtype=torch.int64,device=device),torch.zeros(len(examples),dtype=torch.int64,device=device)] for k in xs}
 return xs,metadata
