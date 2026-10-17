"""Paths, immutable evidence writes and source hashes for the independent audit."""
import datetime,hashlib,json,os,pathlib
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs'
OUT.mkdir(parents=True,exist_ok=True)
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def dump(p,obj):
    p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,allow_nan=False)
def data_root():
    p=pathlib.Path(read(ROOT.parent/'STORAGE_LOCATION_V1.json')['data_root'])/'cycle59'
    p.mkdir(parents=True,exist_ok=True)
    return p
def source_root():return ROOT/'upstream/magnet-tutorial2'
