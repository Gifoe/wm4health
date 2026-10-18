import os
os.environ['OMP_NUM_THREADS']='4'
os.environ['OPENBLAS_NUM_THREADS']='4'
import pathlib,json,hashlib,datetime
ROOT=pathlib.Path(__file__).resolve().parents[1]
PROJECT=ROOT.parent
OUT=ROOT/'outputs'
OUT.mkdir(parents=True,exist_ok=True)
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def blob(b):return hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
def dump(p,obj):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,allow_nan=False)
def state(name,obj):
 p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps({'utc':utc(),**obj},indent=2,allow_nan=False));t.replace(p)
def paths():
 cfg=read(PROJECT/'STORAGE_LOCATION_V1.json')
 d={k:pathlib.Path(cfg[k])/'cycle62' for k in ['data_root','checkpoint_root','artifact_root']}
 for p in d.values():p.mkdir(exist_ok=True,parents=True)
 return d
