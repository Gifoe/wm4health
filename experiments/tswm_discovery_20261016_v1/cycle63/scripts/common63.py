import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['OMP_NUM_THREADS']='4'
os.environ['OPENBLAS_NUM_THREADS']='4'
import pathlib,json,hashlib,datetime
ROOT=pathlib.Path(__file__).resolve().parents[1];PROJECT=ROOT.parent;OUT=ROOT/'outputs';OUT.mkdir(exist_ok=True,parents=True)
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def dump(p,obj):
 p=pathlib.Path(p);p.parent.mkdir(exist_ok=True,parents=True)
 with p.open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,allow_nan=False)
def paths():
 cfg=read(PROJECT/'STORAGE_LOCATION_V1.json');d={k:pathlib.Path(cfg[k])/'cycle63' for k in ['data_root','checkpoint_root','artifact_root']}
 for p in d.values():p.mkdir(exist_ok=True,parents=True)
 return d
def sources():
 s=read(OUT/'SOURCES63_V1.json')
 for r in s['files']:assert sha(r['server_path'])==r['sha256'],r['path']
 return s
