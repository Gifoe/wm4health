"""Pinned explicit-memory and recurrent kernels under a common real-process contract."""
import pathlib,sys,math,os,json,hashlib,time,random,datetime,warnings
R=pathlib.Path(__file__).resolve().parents[1];ROOT=R.parent
sys.path.insert(0,str(ROOT/'cycle29/scripts'));import runtime_profile
sys.path.insert(0,str(R/'vendor'));sys.path.insert(0,str(R/'source_archive/torchdde'));sys.path.insert(0,str(R/'source_archive/learnable_delays'))
import numpy as np
import torch
from torch import nn
from torchdde import integrate,RK2,TorchLinearInterpolator
from learnable_delays.models.models import NODE,NDDE,GRUModel
H=96;FUTURE=48;DT=1/24;SEEDS=[2026481,2026482];CASES=['ett','jena'];METHODS=['node','fixed_dde','learned_dde','gru']
torch.set_num_threads(1)
warnings.filterwarnings('ignore',message='b_err_list is only zeros,.*')
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8'))
def dump(p,x):pathlib.Path(p).write_text(json.dumps(x,indent=2),encoding='utf-8')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def seed(s):random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def checksum(model):
 h=hashlib.sha256()
 for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def clock(t,phase):
 angle=2*math.pi*(phase+t);return torch.stack((angle.sin(),angle.cos()),-1)
def pack(t,signal,phase):return torch.cat((signal,clock(t,phase)),-1)
class MemoryModel(nn.Module):
 def __init__(self,method):
  super().__init__();self.method=method
  if method=='gru':
   self.core=GRUModel(3,32,1);self.core.linear=nn.Linear(32,1);head=self.core.linear
  elif method=='node':self.core=NODE(3,1,32,4);head=next(x for x in reversed(list(self.core.mlp.children())) if isinstance(x,nn.Linear))
  else:
   self.core=NDDE(torch.tensor([.271,1.229]),3,1,32,4);self.core.delays.requires_grad_(method=='learned_dde');head=next(x for x in reversed(list(self.core.mlp.children())) if isinstance(x,nn.Linear))
  nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
 def project(self):
  if self.method=='learned_dde':
   with torch.no_grad():old=self.core.delays.clone();self.core.delays.clamp_(2/24,3.);return int((old!=self.core.delays).sum())
  return 0
 def forward(self,history,phase,horizon=FUTURE,dt=DT):
  assert history.ndim==3 and history.shape[1]==H and history.shape[2]==1
  B=len(history);device=history.device;dtype=history.dtype
  if self.method=='gru':
   times=torch.arange(-H+2,2,device=device,dtype=dtype)/24;inputs=torch.cat((history,clock(times[None,:],phase[:,None])),-1);outputs,hidden=self.core(inputs);previous=history[:,-1];value=previous+outputs[:,-1];values=[value]
   for lead in range(2,horizon+1):
    out,hidden=self.core(pack(history.new_tensor(lead/24),value,phase)[:,None],hidden);value=value+out[:,0];values.append(value)
   return torch.stack(values,1)
  model=self.core
  if self.method=='node':
   def field(t,y,args):return model(t,pack(t,y,args),None)
   initial=history[:,-1];delays=None
  else:
   history_times=torch.arange(-H+1,1,device=device,dtype=dtype)/24;interp=TorchLinearInterpolator(history_times,history);delays=model.delays;initial=interp
   def field(t,y,args,*,history):return model(t,pack(t,y,args),None,history=[pack(t-tau,value,args) for tau,value in zip(model.delays,history)])
  ts=torch.arange(1,horizon+1,device=device,dtype=dtype)/24;solver=RK2();return integrate(field,solver,history.new_tensor(0.),ts[-1],ts,initial,phase,dt0=history.new_tensor(dt),delays=delays,discretize_then_optimize=True,max_steps=4096)
def load(case,split,device='cpu'):
 p=R/'data'/case/(split+'.npz');m=read(R/'data/MANIFEST.json');assert sha(p)==m['files'][case+'/'+split+'.npz']['sha256'];z=np.load(p);return {k:torch.as_tensor(z[k],device=device,dtype=torch.float32) if k in ['history','future','phase'] else z[k] for k in z.files}
