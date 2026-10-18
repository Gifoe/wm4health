import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
import sys,json,hashlib,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];C=ROOT/'cycle87';O=C/'outputs';O.mkdir(parents=True,exist_ok=True)
STORAGE=json.loads((ROOT/'STORAGE_LOCATION_V1.json').read_text(encoding='utf-8-sig'))
RAW=Path(STORAGE['artifact_root'])/'cycle87';RAW.mkdir(parents=True,exist_ok=True)
CKPT=Path(STORAGE['checkpoint_root'])/'cycle87';CKPT.mkdir(parents=True,exist_ok=True)
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,allow_nan=False)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def log(obj):
 obj=dict(utc=utc(),**obj)
 with (O/'trace87_v1.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(obj,allow_nan=False)+'\n')
 print(json.dumps(obj,allow_nan=False),flush=True)
