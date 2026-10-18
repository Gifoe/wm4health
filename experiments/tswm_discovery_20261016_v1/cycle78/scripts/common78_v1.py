import os,pathlib,json,hashlib,datetime
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2';os.environ['CUDA_VISIBLE_DEVICES']='-1'
ROOT=pathlib.Path(__file__).resolve().parents[1];PROJECT=ROOT.parent;OUT=ROOT/'outputs';OUT.mkdir(parents=True,exist_ok=True)
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def state(name,x):
 p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(dict(utc=utc(),**x),indent=2,allow_nan=False));t.replace(p)
def storage(kind):
 p=pathlib.Path(read(PROJECT/'STORAGE_LOCATION_V1.json')[kind+'_root'])/'cycle78';assert p.drive.upper()=='G:';p.mkdir(parents=True,exist_ok=True);return p
