"""Acquire one original lighting module and the official OBJ-reader definition."""
import pathlib,json,hashlib,urllib.request,base64,datetime,traceback
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'renderer_source_supplement_v1';OUT.mkdir(exist_ok=False)
G=pathlib.Path(r'G:\ICLR Omni-eeg cache\tswm_discovery_20261016_v1\caches');GRAD=G/'gradsim_abe8008_native_renderer_v1';K=G/'kaolin_0_17_obj_reader_v1';K.mkdir(exist_ok=False)
def get(url):
 with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'native-renderer-boundary-audit'}),timeout=25) as f:return json.load(f)
def dump(n,x):
 with (OUT/n).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2)
def acquire(repo,pin,rel,base):
 j=get('https://api.github.com/repos/'+repo+'/contents/'+rel+'?ref='+pin);assert j['encoding']=='base64';b=base64.b64decode(j['content']);assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==j['sha']
 p=base/rel;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(b)
 return dict(repo=repo,revision=pin,path=rel,git_blob_sha=j['sha'],sha256=hashlib.sha256(b).hexdigest(),bytes=len(b),cache_path=str(p))
try:
 rows=[acquire('gradsim/gradsim','abe8008e9559957f84a67b06208600cb66de3be2','gradsim/renderutils/lighting.py',GRAD)]
 pin=get('https://api.github.com/repos/NVIDIAGameWorks/kaolin/commits/v0.17.0')['sha']
 rows.append(acquire('NVIDIAGameWorks/kaolin',pin,'kaolin/io/obj.py',K))
 dump('RESULTS_V1.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=rows,model_evaluations=0,new_fits=0,local_payload_sync=False));print(json.dumps(dict(status='NATIVE_LIGHTING_AND_OFFICIAL_OBJ_READER_ACQUIRED',files=rows)),flush=True)
except Exception as e:
 dump('FAILURE_V1.json',dict(exception_type=type(e).__name__,exception=str(e),traceback=traceback.format_exc()));raise
