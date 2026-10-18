"""Inference dependencies into an isolated prefix, never change shared site-packages."""
import sys,pathlib,subprocess,importlib.metadata
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common80_v1 import *
target=storage('cache')/'python310_dependencies_v2';assert not target.exists(),'Do not silently overwrite a dependency prefix';target.mkdir()
os.environ['PIP_CACHE_DIR']=str(storage('cache')/'pip_cache_v2')
packages=['transformers==4.56.2','tokenizers==0.22.1','peft==0.17.1','optree==0.17.0','accelerate==1.10.1','huggingface_hub==0.36.2','regex==2025.9.18']
log=OUT/'dependency_install80_v2.log'
command=[sys.executable,'-m','pip','install','--disable-pip-version-check','--no-input','--no-deps','--index-url','https://pypi.org/simple','--target',str(target),*packages]
with log.open('xb') as f:result=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT)
dump(OUT/'DEPENDENCY_INSTALL80_V2.json',dict(utc=utc(),target=str(target),command=command,exit_code=result.returncode,log_sha256=sha(log),shared_environment_modified=False))
assert result.returncode==0,result.returncode
print(json.dumps(dict(status='ISOLATED_PREFIX_INSTALLED',target=str(target),packages=packages,exit_code=result.returncode)),flush=True)
