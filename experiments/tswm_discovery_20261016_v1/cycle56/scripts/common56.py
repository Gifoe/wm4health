"""Remote-only paths, source pins and exclusive result writes."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ['JAX_PLATFORM_NAME']='cpu'
os.environ['JAX_ENABLE_X64']='true'
if os.name=='nt':
    import ctypes
    fn=ctypes.windll.kernel32.SetProcessAffinityMask
    fn.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
    assert fn(ctypes.c_void_p(-1),ctypes.c_size_t(0xf))
import datetime,hashlib,json,pathlib,sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
PROJECT=ROOT.parent;OUT=ROOT/'outputs';OUT.mkdir(parents=True,exist_ok=True)
SOURCE=PROJECT/'upstream/rp-gssm'
sys.path.insert(0,str(SOURCE))
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
def dump(p,x):
    p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def source_check():
    old=read(PROJECT/'cycle54/plan54_v1.json')
    assert old['source_revision']=='a75225d5edceb972bb872dc9a335608733567dd4'
    actual={p:sha(SOURCE/p) for p in old['source_files_sha256']}
    assert actual==old['source_files_sha256']
    return actual
