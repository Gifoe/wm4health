"""Direct compute-host acquisition of pinned public source and selected checkpoints."""
import pathlib,json,hashlib,urllib.request,datetime,concurrent.futures,threading,time
ROOT=pathlib.Path(__file__).resolve().parents[1];PIN='3d47140f18262628d252858c3efdfe2ad1e21a86';REPO='GoForIt-YYB/Meta-iLaD';OUT=ROOT/'acquisition_v1';OUT.mkdir(exist_ok=False);lock=threading.Lock()
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
def sha(b):return hashlib.sha256(b).hexdigest()
def get(url,limit):
 for attempt in range(3):
  try:
   with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'native-research-provenance-audit'}),timeout=30) as r:b=r.read(limit+1)
   assert len(b)<=limit;return b
  except Exception:
   if attempt==2:raise
   time.sleep(1+attempt)
def event(stage,**kw):
 x=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),stage=stage,**kw)
 with lock:
  with (OUT/'trace.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(x)+'\n')
  print(json.dumps(x),flush=True)
def dump(n,x):
 with (OUT/n).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
try:
 storage=read(ROOT.parent/'STORAGE_LOCATION_V1.json');cache=pathlib.Path(storage['cache_root'])/ROOT.name;SRC=cache/'source';SRC.mkdir(parents=True,exist_ok=False)
 api='https://api.github.com/repos/'+REPO;commit=json.loads(get(api+'/commits/'+PIN,2000000));assert commit['sha']==PIN;tree=json.loads(get(api+'/git/trees/'+PIN+'?recursive=1',800000));assert not tree['truncated']
 blobs=[x for x in tree['tree'] if x['type']=='blob'];wanted=[]
 for x in blobs:
  p=x['path'];assert '..' not in p.split('/') and not p.startswith('/')
  if p in ['README.md','LICENSE','THIRD_PARTY_NOTICES.md'] or (p.startswith('mocap/') and (pathlib.PurePosixPath(p).suffix in ['.py','.md','.txt','.json','.sh','.pt'])):wanted.append(x)
 weights=[x for x in wanted if x['path'].endswith('.pt')];assert len(weights)==9 and all(x['size']<4000000 for x in weights)
 plan=dict(source_repo='https://github.com/'+REPO,source_revision=PIN,commit_utc=commit['commit']['committer']['date'],source_root=str(SRC),repository_blobs=len(blobs),repository_blob_bytes=sum(x.get('size',0) for x in blobs),selected_blobs=len(wanted),selected_bytes=sum(x['size'] for x in wanted),selected_checkpoints=len(weights),scope='MoCAP refined native release only',selected_files=wanted,source_documents_inspected_before_execution=False,local_payload_sync=False)
 dump('PLAN_V1.json',plan);event('acquisition_plan_frozen',files=len(wanted),bytes=plan['selected_bytes'])
 def acquire(x):
  p=SRC/pathlib.PurePosixPath(x['path']);p.parent.mkdir(parents=True,exist_ok=True);b=get('https://raw.githubusercontent.com/'+REPO+'/'+PIN+'/'+x['path'],max(1000000,x['size']+1));assert len(b)==x['size'];git=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest();assert git==x['sha']
  with p.open('xb') as f:f.write(b)
  assert sha(p.read_bytes())==sha(b);row=dict(path=x['path'],bytes=len(b),git_blob_sha=git,sha256=sha(b),checkpoint=x['path'].endswith('.pt'));event('file_sha_verified',**row);return row
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:files=list(pool.map(acquire,wanted))
 result=dict(status='PINNED_NATIVE_MOCAP_SOURCE_AND_NINE_CHECKPOINTS_ACQUIRED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_repo=plan['source_repo'],source_revision=PIN,source_root=str(SRC),files=files,source_documents_inspected_before_execution=False,raw_signal_data_downloaded=False,scientific_fits=0,evaluations=0,local_payload_sync=False,protocol_sha256=sha((ROOT/'ACQUISITION_AND_REPRODUCTION_SCOPE_V1.md').read_bytes()))
 dump('RESULTS_V1.json',result);event('native_acquisition_complete',files=len(files),checkpoints=len(weights),bytes=sum(x['bytes'] for x in files))
except Exception as e:
 import traceback
 dump('FAILURE_V1.json',dict(exception_type=type(e).__name__,exception=str(e),traceback=traceback.format_exc()));raise
