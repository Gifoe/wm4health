"""Official demo C/H sizes with generated data, not unavailable demo data."""
import sys,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent/'cycle29/scripts'))
import runtime_profile
import json,hashlib,importlib,types,inspect,time,random
import numpy as np
import torch
from safetensors.torch import load_file
UP=ROOT/'upstream/normwear2';Q=ROOT/'outputs/normwear_qualification_v3'
OUT=ROOT/'outputs/normwear_demo_extent_v1'
assert not OUT.exists();OUT.mkdir(parents=True)
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
for r in json.loads((UP/'source_manifest.json').read_text())['source_files']:assert sha(UP/r['path'])==r['sha256']
assert json.loads((Q/'verification.json').read_text())['status']=='passed'
pkg=types.ModuleType('extent_normwear31');pkg.__path__=[str(UP)];sys.modules[pkg.__name__]=pkg
mod=importlib.import_module(pkg.__name__+'.normwear2');lb=importlib.import_module(pkg.__name__+'.latent_bayesian')
cfg=json.loads((UP/'config.json').read_text());args={k:v for k,v in cfg.items() if k in inspect.signature(mod.NormWear2.__init__).parameters}
torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
model=mod.NormWear2(**args);s=load_file(str(UP/'model.safetensors'));model.load_state_dict(s,strict=True);del s
model.eval().to('cuda');C=H=2048
t=torch.arange(C+H,device='cuda',dtype=torch.float32)
x=torch.stack((torch.sin(t*.017)+.13*torch.cos(t*.059),torch.cos(t*.012)+.2*torch.sin(t*.041),
               .2*torch.sin(t*.009)+.3*torch.cos(t*.023)),0)[None]
loc=x[:,:,:C].mean(-1,keepdim=True);scale=x[:,:,:C].std(-1,keepdim=True)+1e-8
x=((x-loc)/scale).clamp(-3,3);x[:,:2,C:]=0
kw=dict(n_channels=3,physio_channels=[0,1],context_length=C,pred_length=H,
        num_states=int(np.log((C//16)**2)+.5),action_channels=[2],condition_bayes=True,
        num_traj_sampled=1,latent_encoder=model,return_transit_matrix=False)
def seed():random.seed(20263102);np.random.seed(20263102);torch.manual_seed(20263102)
report={'context':C,'horizon':H,'num_states':kw['num_states'],'identity':'official demo sizes/interface, generated data; not original private dataset or scientific performance',
        'source_revision':json.loads((UP/'source_manifest.json').read_text())['revision'],'script_sha256':sha(__file__)}
original=lb.bayesian_forecast
seed();start=time.perf_counter()
with torch.no_grad():
 try:original(x,**kw)
 except IndexError as exc:report['native']={'status':'IndexError','error':repr(exc)}
 else:raise AssertionError('native length error did not replicate at official demo extent')
report['native']['seconds']=time.perf_counter()-start
namespace={};repair=(Q/'ordinary_patch_count_repair.py').read_text(encoding='utf-8')
exec(compile(repair,'ordinary_patch_count_repair.py','exec'),lb.__dict__,namespace)
seed();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
with torch.no_grad():fixed=namespace['bayesian_forecast'](x,**kw)
torch.cuda.synchronize();assert fixed.shape==(2,H) and torch.isfinite(fixed).all()
report['length_repaired']={'status':'passed','shape':list(fixed.shape),'seconds':time.perf_counter()-start,
    'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'action_index':'original, not repaired in this isolated length check'}
np.savez_compressed(OUT/'arrays.npz',input=x.cpu().numpy(),ordinary_length_output=fixed.cpu().numpy())
report['arrays_sha256']=sha(OUT/'arrays.npz');report['qualification_sha256']=sha(Q/'qualification.json')
(OUT/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report),flush=True)
