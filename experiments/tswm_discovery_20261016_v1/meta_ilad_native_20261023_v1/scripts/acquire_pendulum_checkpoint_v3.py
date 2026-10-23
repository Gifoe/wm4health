"""Acquire one native physical checkpoint and its actual evaluation definitions."""
import pathlib,json,hashlib,urllib.request,base64,datetime,concurrent.futures,traceback
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'pendulum_acquisition_recovery_v2';OUT.mkdir(exist_ok=False)
SRC=pathlib.Path(r'G:\ICLR Omni-eeg cache\tswm_discovery_20261016_v1\caches\meta_ilad_native_20261023_v1\source')
PIN='3d47140f18262628d252858c3efdfe2ad1e21a86';B='https://api.github.com/repos/GoForIt-YYB/Meta-iLaD/'
def dump(n,x):
 with (OUT/n).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2)
def get(p):
 with urllib.request.urlopen(urllib.request.Request(B+p,headers={'User-Agent':'pinned-native-physics-checkpoint'}),timeout=45) as f:return json.load(f)
try:
 old=json.loads((ROOT/'pendulum_acquisition_v1/PLAN_V1.json').read_text());assert old['source_revision']==PIN;chosen=old['selected']
 dump('PLAN_V1.json',dict(original_plan=old,recovery='public CDN transport for six missing source files; original Git blob verification required; both prior failures preserved'))
 def acquire(x):
  p=SRC/pathlib.PurePosixPath(x['path']);p.parent.mkdir(parents=True,exist_ok=True)
  if p.exists():b=p.read_bytes()
  else:
   url='https://cdn.jsdelivr.net/gh/GoForIt-YYB/Meta-iLaD@'+PIN+'/'+x['path']
   with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'pinned-missing-source-recovery'}),timeout=25) as f:b=f.read(x['size']+1)
  assert len(b)==x['size'] and hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==x['sha']
  if not p.exists():
   with p.open('xb') as f:f.write(b)
  return dict(path=x['path'],bytes=len(b),git_blob_sha=x['sha'],sha256=hashlib.sha256(b).hexdigest())
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(acquire,chosen))
 dump('RESULTS_V1.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_revision=PIN,source_root=str(SRC),files=rows,model_evaluations=0,new_fits=0,local_payload_sync=False));print(json.dumps(dict(status='PINNED_PHYSICAL_CHECKPOINT_AND_EVALUATORS_ACQUIRED',files=len(rows),bytes=sum(x['bytes'] for x in rows))),flush=True)
except Exception as e:
 dump('FAILURE_V1.json',dict(exception_type=type(e).__name__,exception=str(e),traceback=traceback.format_exc()));raise
