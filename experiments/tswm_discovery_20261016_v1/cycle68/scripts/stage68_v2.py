import sys,pathlib
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common68_v1 import *
import urllib.request,zipfile
rev='4c20d94ccab51abf4ebf68aa8810f4596381dd41';base=storage('checkpoint')/'sysid_meta';base.mkdir(exist_ok=True)
files=[('transformer_sim.py',19581,'bffa3b730535ca3d32d21ae2963060f132b591bd'),('README.md',3944,'9db47032094a316023093021cf798061db23b895'),('train_sim_WH_skip.py',14838,'9651dcb8e3cfdf832f23e10bc4553d3e5da4240e'),('benchmarks.ipynb',456250,'aeb976d3a38ac417f4cf6d7aa8700f67eec3fc64')];rows=[]
for name,size,oid in files+[('ckpt_430_fine-tuned_3100000_lower_noise.pt',90193824,None)]:
 url=('https://raw.githubusercontent.com/PuraFilippo/sysid_benchmarks/'+rev+'/'+name) if oid else ('https://github.com/PuraFilippo/sysid_benchmarks/releases/download/v0.1/'+name)
 p=base/name;part=base/(name+'.part')
 if not p.exists():
  assert not part.exists(),'Preserve partial and inspect; no blind restart'
  n=0
  with urllib.request.urlopen(url,timeout=45) as inp,part.open('xb') as out:
   while b:=inp.read(1024*1024):
    out.write(b);n+=len(b);assert n<=size;state('stage_state68_v1.json',dict(status='DOWNLOADING',file=name,received=n,expected=size))
  assert n==size
  if oid:
   raw=part.read_bytes();assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==oid
  else:
   with zipfile.ZipFile(part) as z:assert z.testzip() is None
  part.rename(p)
 assert p.stat().st_size==size
 if oid:
  raw=p.read_bytes();assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==oid
 rows.append(dict(name=name,path=str(p),bytes=size,source_git_blob=oid,sha256=sha(p),url=url))
dump(OUT/'MODEL_FILES68_V1.json',dict(utc=utc(),status='SOURCE_GIT_BLOBS_AND_RELEASE_ZIP_CRC_VERIFIED_SERVER_SHA_RECORDED',source_repo='PuraFilippo/sysid_benchmarks',revision=rev,release_tag='v0.1',upstream_weight_sha256_published=False,files=rows,server_directory=str(base),no_local_results_synced=True,third_party_license='No LICENSE file present in pinned repository tree; no payload redistribution'))
state('stage_state68_v1.json',dict(status='SOURCE_STAGE_TERMINAL_SUCCESS',files=len(rows)));print('SOURCE_STAGE_TERMINAL_SUCCESS',flush=True)
