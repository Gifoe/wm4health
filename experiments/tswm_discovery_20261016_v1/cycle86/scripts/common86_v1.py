import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import ast,json,hashlib,datetime,sys,types
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(2)
ROOT=Path(__file__).resolve().parents[2];C=ROOT/'cycle86';O=C/'outputs';O.mkdir(parents=True,exist_ok=True)
CONFIG=json.loads((ROOT/'STORAGE_LOCATION_V1.json').read_text(encoding='utf-8-sig'))
RAW=Path(CONFIG['artifact_root'])/'cycle86'/'sampling_v1';RAW.mkdir(parents=True,exist_ok=True)
CACHE=Path(CONFIG['cache_root'])/'cycle80'/'OpenFIM_cee2bb53_v1'
SOURCE=CACHE/'src'/'fim'/'data_generation'/'sde'/'dynamical_systems_sample.py'
SEEDS=[2026861,2026862,2026863]
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,allow_nan=False)
def save(p,**arrays):
    with Path(p).open('xb') as f:np.savez_compressed(f,**arrays)
def log(obj):
    obj=dict(utc=utc(),**obj)
    with (O/'trace86_v1.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(obj,allow_nan=False)+'\n')
    print(json.dumps(obj,allow_nan=False),flush=True)
