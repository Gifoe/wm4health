"""Immutable receipts; data and weights stay on the authorized G disk."""
import os
os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='4'
import pathlib,json,hashlib,datetime
ROOT=pathlib.Path(__file__).resolve().parents[1];PROJECT=ROOT.parent;OUT=ROOT/'outputs';OUT.mkdir(exist_ok=True,parents=True)
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def dump(p,o):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f:json.dump(o,f,indent=2,allow_nan=False)
def paths():
 cfg=read(PROJECT/'STORAGE_LOCATION_V1.json');d={k:pathlib.Path(cfg[k])/'cycle61' for k in ['data_root','checkpoint_root','artifact_root']}
 for p in d.values():p.mkdir(parents=True,exist_ok=True)
 return d
def sources():
 o=read(OUT/'SOURCES61_V1.json')
 for r in o['files']:assert sha(r['server_path'])==r['sha256'],r['path']
 return o
