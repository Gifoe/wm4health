"""Acquire bounded, pinned generator source on the remote authorized data disk."""
import pathlib,json,hashlib,urllib.request,base64,datetime,concurrent.futures,traceback
ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'physics_qualification_v1';OUT.mkdir(exist_ok=False)
PIN='3d47140f18262628d252858c3efdfe2ad1e21a86'
BASE='https://api.github.com/repos/GoForIt-YYB/Meta-iLaD/'
SRC=pathlib.Path(r'G:\ICLR Omni-eeg cache\tswm_discovery_20261016_v1\caches\meta_ilad_native_20261023_v1\source')
def sha(b):return hashlib.sha256(b).hexdigest()
def dump(n,x):
 with (OUT/n).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2)
def get(path):
 with urllib.request.urlopen(urllib.request.Request(BASE+path,headers={'User-Agent':'pinned-physics-qualification'}),timeout=25) as f:return json.load(f)
try:
 tree=get('git/trees/'+PIN+'?recursive=1');assert not tree['truncated']
 paths=['pendulum/README.md','pendulum/gen_data/dynamics.py','pendulum/gen_data/generate_data.py','pendulum/gen_data/fingerprint.py','pendulum/gen_data/fingerprints.json','pendulum/gen_data/renderer.py','pendulum/gen_data/requirements.txt','pendulum/gen_data/sampledata/sphere.obj']
 chosen=[x for x in tree['tree'] if x['path'] in paths];assert len(chosen)==len(paths)
 dump('PLAN_V1.json',dict(source_revision=PIN,selected=chosen,source_root=str(SRC),protocol_sha256=sha((ROOT/'PHYSICS_QUALIFICATION_PROTOCOL_V1.md').read_bytes()),model_evaluations=0))
 def acquire(x):
  p=SRC/pathlib.PurePosixPath(x['path']);p.parent.mkdir(parents=True,exist_ok=True)
  if p.exists():b=p.read_bytes()
  else:
   j=get('git/blobs/'+x['sha']);assert j['encoding']=='base64';b=base64.b64decode(j['content'])
  assert len(b)==x['size'] and hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==x['sha']
  if not p.exists():
   with p.open('xb') as f:f.write(b)
  return dict(path=x['path'],bytes=len(b),git_blob_sha=x['sha'],sha256=sha(b))
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(acquire,chosen))
 dump('ACQUISITION_V1.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_revision=PIN,files=rows,model_evaluations=0,new_fits=0,local_payload_sync=False))
 print(json.dumps(dict(status='PHYSICS_GENERATOR_SOURCE_ACQUIRED',files=len(rows),bytes=sum(x['bytes'] for x in rows))),flush=True)
 print('CHECKER_SOURCE\n'+(SRC/'pendulum/gen_data/fingerprint.py').read_text(encoding='utf-8'))
except Exception as e:
 dump('ACQUISITION_FAILURE_V1.json',dict(exception_type=type(e).__name__,exception=str(e),traceback=traceback.format_exc()));raise
