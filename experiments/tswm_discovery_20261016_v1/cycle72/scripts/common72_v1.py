import os,sys,pathlib,json,hashlib,datetime
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['MPLBACKEND']='Agg'
ROOT=pathlib.Path(__file__).resolve().parents[1];PROJECT=ROOT.parent;OUT=ROOT/'outputs';OUT.mkdir(exist_ok=True,parents=True)
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(path):return json.loads(pathlib.Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
def dump(path,obj,exclusive=True):
    path=pathlib.Path(path);path.parent.mkdir(exist_ok=True,parents=True)
    with path.open('x' if exclusive else 'w',encoding='utf-8') as f:json.dump(obj,f,indent=2,allow_nan=False)
def storage(kind):
    path=pathlib.Path(read(PROJECT/'STORAGE_LOCATION_V1.json')[kind+'_root'])/'cycle72'
    assert path.drive.upper()=='G:';path.mkdir(exist_ok=True,parents=True);return path
sys.path.insert(0,str(PROJECT/'cycle71'/'scripts'))
from runtime_affinity71_v1 import configure
AFFINITY=configure('efficient')
sys.path.remove(str(PROJECT/'cycle71'/'scripts'))
