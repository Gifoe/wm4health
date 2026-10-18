import os,sys,json,datetime,hashlib
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
ROOT=Path(__file__).resolve().parents[2];C=ROOT/'cycle90';O=C/'outputs';O.mkdir(parents=True,exist_ok=True)
STORAGE=json.loads((ROOT/'STORAGE_LOCATION_V1.json').read_text(encoding='utf-8-sig'))
sys.path.insert(0,str(ROOT/'cycle89/scripts'))
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def log(x):
 x=dict(utc=utc(),**x)
 with (O/'trace90_v1.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(x,allow_nan=False)+'\n')
 print(json.dumps(x,allow_nan=False),flush=True)
