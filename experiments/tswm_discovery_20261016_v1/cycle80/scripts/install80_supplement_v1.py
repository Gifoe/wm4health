"""Add only absent optional import dependencies to C80's prefix before numerical runs."""
import sys,pathlib,subprocess
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common80_v1 import *
info=read(OUT/'DEPENDENCY_INSTALL80_V2.json');target=pathlib.Path(info['target']);assert info['exit_code']==0 and target.is_dir()
for name in ['torchsde','trampoline']:assert not (target/name).exists()
os.environ['PIP_CACHE_DIR']=str(storage('cache')/'pip_cache_v2');packages=['torchsde==0.2.6','trampoline==0.1.2'];log=OUT/'dependency_supplement80_v1.log'
command=[sys.executable,'-m','pip','install','--disable-pip-version-check','--no-input','--no-deps','--index-url','https://pypi.org/simple','--target',str(target),*packages]
with log.open('xb') as f:r=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT)
dump(OUT/'DEPENDENCY_SUPPLEMENT80_V1.json',dict(utc=utc(),target=str(target),packages=packages,exit_code=r.returncode,log_sha256=sha(log),existing_package_files_overwritten=False,shared_environment_modified=False));assert r.returncode==0
print(json.dumps(dict(status='ABSENT_IMPORT_DEPENDENCIES_ADDED',packages=packages)),flush=True)
