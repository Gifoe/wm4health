"""Acquire only the pinned native forward renderer and license on server G."""
import pathlib,sys,json,hashlib,datetime,urllib.request,base64,concurrent.futures,traceback
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'renderer_acquisition_v1';OUT.mkdir(exist_ok=False)
SRC=pathlib.Path(r'G:\ICLR Omni-eeg cache\tswm_discovery_20261016_v1\caches\gradsim_abe8008_native_renderer_v1');SRC.mkdir(exist_ok=False)
PIN='abe8008e9559957f84a67b06208600cb66de3be2';B='https://api.github.com/repos/gradsim/gradsim/'
def sha(b):return hashlib.sha256(b).hexdigest()
def dump(n,x):
 with (OUT/n).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2)
def get(p):
 with urllib.request.urlopen(urllib.request.Request(B+p,headers={'User-Agent':'pinned-native-renderer-qualification'}),timeout=25) as f:return json.load(f)
try:
 t=get('git/trees/'+PIN+'?recursive=1');assert not t['truncated']
 paths=['setup.py','gradsim/renderutils/softrenderer.py','gradsim/renderutils/soft_rasterize.py','gradsim/renderutils/cuda/soft_rasterize_cuda_kernel.cu','gradsim/renderutils/cuda/soft_rasterize_cuda.cpp','gradsim/utils/meshutils.py']
 wanted=[x for x in t['tree'] if x['type']=='blob' and (x['path'] in paths or (x['path'].count('/')==0 and 'license' in x['path'].lower()))]
 assert len(wanted)>=len(paths)
 dump('PLAN_V1.json',dict(source_revision=PIN,source_repo='https://github.com/gradsim/gradsim',selected=wanted,source_root=str(SRC),protocol_sha256=sha((ROOT/'NATIVE_RENDERER_FEASIBILITY_PROTOCOL_V1.md').read_bytes())))
 def acquire(x):
  j=get('git/blobs/'+x['sha']);b=base64.b64decode(j['content']);assert len(b)==x['size'] and hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==x['sha']
  p=SRC/pathlib.PurePosixPath(x['path']);p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb') as f:f.write(b)
  return dict(path=x['path'],git_blob_sha=x['sha'],sha256=sha(b),bytes=len(b))
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(acquire,wanted))
 dump('RESULTS_V1.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_revision=PIN,source_root=str(SRC),files=rows,model_evaluations=0,new_fits=0,local_payload_sync=False))
 print(json.dumps(dict(status='PINNED_FORWARD_RENDERER_SOURCE_ACQUIRED',files=len(rows),bytes=sum(x['bytes'] for x in rows))),flush=True)
except Exception as e:
 dump('FAILURE_V1.json',dict(exception_type=type(e).__name__,exception=str(e),traceback=traceback.format_exc()));raise
