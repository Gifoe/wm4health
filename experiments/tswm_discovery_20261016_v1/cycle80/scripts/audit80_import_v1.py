"""Record full-package import before considering an inference-only package registration wrapper."""
import sys,pathlib,traceback,importlib.metadata
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common80_v1 import *
sys.path.insert(0,str(PROJECT/'cycle71/scripts'))
from runtime_affinity71_v1 import configure
AFFINITY=configure('efficient')
m=read(OUT/'SOURCE_MANIFEST80_V1.json');source=pathlib.Path(m['source_root'])
for item in m['files']:assert sha(source/item['path'])==item['sha256']
deps=read(OUT/'DEPENDENCY_INSTALL80_V1.json');assert deps['exit_code']==0
sys.path.insert(0,deps['target']);sys.path.insert(0,str(source/'src'))
status='PASS_NATIVE_FULL_PACKAGE_IMPORT';error=None
try:
 import torch
 torch.set_num_threads(2)
 from fim.models.ode import load_fim_ode_local
except Exception:
 status='ENGINEERING_FULL_PACKAGE_IMPORT_FAILURE';error=traceback.format_exc()
versions={}
for name in ['torch','numpy','transformers','tokenizers','peft','optree','accelerate','huggingface_hub','safetensors','torchinfo','scipy']:
 try:versions[name]=importlib.metadata.version(name)
 except importlib.metadata.PackageNotFoundError:versions[name]=None
dump(OUT/'IMPORT_AUDIT80_V1.json',dict(utc=utc(),status=status,error=error,versions=versions,source_files_checked=len(m['files']),affinity=AFFINITY,numerical_sources_modified=False))
print(json.dumps(dict(status=status,error=error,versions=versions)),flush=True)
