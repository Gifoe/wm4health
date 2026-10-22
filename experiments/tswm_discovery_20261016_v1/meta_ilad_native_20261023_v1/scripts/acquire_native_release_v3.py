"""Recover a terminal raw-endpoint failure through the same pinned Git blobs."""
import pathlib,json,hashlib,urllib.request,datetime,concurrent.futures,threading,time,base64
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'acquisition_recovery_v1';OUT.mkdir(exist_ok=False);lock=threading.Lock();PIN='3d47140f18262628d252858c3efdfe2ad1e21a86';REPO='GoForIt-YYB/Meta-iLaD'
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
def sha(b):return hashlib.sha256(b).hexdigest()
def gitsha(b):return hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
def dump(n,x):
 with (OUT/n).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def event(stage,**kw):
 row=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),stage=stage,**kw)
 with lock:
  with (OUT/'trace.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
  print(json.dumps(row),flush=True)
try:
 old=read(ROOT/'acquisition_v1/PLAN_V1.json');assert old['source_revision']==PIN and old['source_repo']=='https://github.com/'+REPO;assert read(ROOT/'acquisition_execution_v1/exit.json')['exit_code']==1
 SRC=pathlib.Path(old['source_root']);assert SRC.is_dir();wanted=old['selected_files'];assert len(wanted)==50;dump('PLAN_V1.json',dict(original_plan=old,transport='api.github.com Git blob base64',protocol_sha256=sha((ROOT/'TRANSPORT_RECOVERY_V1.md').read_bytes()),raw_endpoint_job_terminal=True))
 def acquire(x):
  assert '..' not in x['path'].split('/') and not x['path'].startswith('/');path=SRC/pathlib.PurePosixPath(x['path']);path.parent.mkdir(parents=True,exist_ok=True);reused=path.exists()
  if reused:b=path.read_bytes()
  else:
   url='https://api.github.com/repos/'+REPO+'/git/blobs/'+x['sha']
   for attempt in range(3):
    try:
     with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'pinned-native-source-recovery','Accept':'application/vnd.github+json'}),timeout=30) as f:raw=f.read(12000001)
     assert len(raw)<=12000000;j=json.loads(raw);assert j['sha']==x['sha'] and j['encoding']=='base64';b=base64.b64decode(j['content']);break
    except Exception:
     if attempt==2:raise
     time.sleep(1+attempt)
  assert len(b)==x['size'] and gitsha(b)==x['sha'],x['path']
  if not reused:
   with path.open('xb') as f:f.write(b)
  assert sha(path.read_bytes())==sha(b);row=dict(path=x['path'],bytes=len(b),git_blob_sha=x['sha'],sha256=sha(b),checkpoint=x['path'].endswith('.pt'),reused_verified_file=reused);event('file_sha_verified',**row);return row
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:files=list(pool.map(acquire,wanted))
 result=dict(status='PINNED_NATIVE_MOCAP_SOURCE_AND_NINE_CHECKPOINTS_ACQUIRED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_repo=old['source_repo'],source_revision=PIN,source_root=str(SRC),files=files,source_documents_inspected_before_execution=False,raw_signal_data_downloaded=False,scientific_fits=0,evaluations=0,local_payload_sync=False,transport='GitHub Git blob API; raw-endpoint failure preserved',protocol_sha256=sha((ROOT/'ACQUISITION_AND_REPRODUCTION_SCOPE_V1.md').read_bytes()));dump('RESULTS_V1.json',result);event('native_acquisition_complete',files=len(files),checkpoints=sum(x['checkpoint'] for x in files),bytes=sum(x['bytes'] for x in files))
except Exception as e:
 import traceback
 dump('FAILURE_V1.json',dict(exception_type=type(e).__name__,exception=str(e),traceback=traceback.format_exc()));raise
