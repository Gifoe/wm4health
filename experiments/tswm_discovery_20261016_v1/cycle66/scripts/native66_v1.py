"""Load all exact pinned native state, no HF wrapper dependency or model proxy."""
import os
os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='4'
import pathlib,json,hashlib,datetime,sys,types,importlib,inspect
import numpy as np,torch
from safetensors.torch import load_file
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs'
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  while b:=f.read(8*1024*1024):h.update(b)
 return h.hexdigest()
def dump(p,obj):
 with p.open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,allow_nan=False)
def load(device='cuda'):
 assert sys.flags.isolated
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest');torch.manual_seed(2021);np.random.seed(2021)
 receipt=read(OUT/'MODEL_FILES66_V1.json');base=pathlib.Path(receipt['server_directory'])
 for row in receipt['files']:assert sha(row['server_path'])==row['sha256']
 pkg=types.ModuleType('normwear66_native');pkg.__path__=[str(base)];sys.modules[pkg.__name__]=pkg;mod=importlib.import_module(pkg.__name__+'.normwear2')
 cfg=read(base/'config.json');args={k:v for k,v in cfg.items() if k in inspect.signature(mod.NormWear2).parameters};model=mod.NormWear2(**args)
 state=load_file(str(base/'model.safetensors'),device='cpu');assert set(state)==set(model.state_dict());model.load_state_dict(state,strict=True);assert all(torch.equal(model.state_dict()[k],v) for k,v in state.items());del state
 return model.to(device).eval(),mod,receipt
def artifacts():
 p=pathlib.Path(read(ROOT.parent/'STORAGE_LOCATION_V1.json')['artifact_root'])/'cycle66';p.mkdir(exist_ok=True,parents=True);return p
