import os
os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='4'
import pathlib,json,hashlib,datetime
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];PROJECT=ROOT.parent;OUT=ROOT/'outputs';OUT.mkdir(exist_ok=True,parents=True)
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def dump(p,obj):
 p=pathlib.Path(p);p.parent.mkdir(exist_ok=True,parents=True)
 with p.open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,allow_nan=False)
def state(name,obj):
 p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(dict(utc=utc(),**obj),indent=2,allow_nan=False),encoding='utf-8');t.replace(p)
def storage(kind):
 p=pathlib.Path(read(PROJECT/'STORAGE_LOCATION_V1.json')[kind+'_root'])/'cycle69';assert p.drive.upper()=='G:';p.mkdir(exist_ok=True,parents=True);return p
