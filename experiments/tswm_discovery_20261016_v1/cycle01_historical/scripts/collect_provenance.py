import datetime,hashlib,importlib.metadata,json,pathlib,platform,subprocess,sys
import numpy as np
import torch
ROOT=pathlib.Path(__file__).resolve().parents[1]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
packages={}
for p in ['torch','numpy','scipy','pandas','matplotlib','reformer-pytorch','einops','scikit-learn']:
    try:packages[p]=importlib.metadata.version(p)
    except importlib.metadata.PackageNotFoundError:packages[p]='missing'
files=[]
for folder in ['scripts','tests']:
    for p in sorted((ROOT/folder).rglob('*.py')):
        files.append({'path':p.relative_to(ROOT).as_posix(),'sha256':digest(p),'bytes':p.stat().st_size})
artifacts=[]
for p in sorted((ROOT/'outputs').rglob('*')):
    if p.is_file() and p.suffix.lower() in ['.pt','.npz']:
        artifacts.append({'path':p.relative_to(ROOT).as_posix(),'sha256':digest(p),'bytes':p.stat().st_size})
gpu=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True)
result={'recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'python':platform.python_version(),'platform':platform.platform(),'packages':packages,
        'cuda_runtime':torch.version.cuda,'gpu':gpu.stdout.strip(),'code':files,
        'config_sha256':digest(ROOT/'config.json'),'large_artifacts_retained_on_server':artifacts,
        'artifact_bytes':sum(p['bytes'] for p in artifacts),'no_publication':True}
(ROOT/'outputs/provenance.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ['code','large_artifacts_retained_on_server']},indent=2))
