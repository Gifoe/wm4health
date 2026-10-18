"""Pin a small MIT author-source selection; no model execution or data download."""
import sys,pathlib,urllib.request,concurrent.futures
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common71_v1 import *

selection=read(ROOT/'SOURCE_SELECTION71_LEJEPA_V1.json');assert selection['revision']=='de7503f1b2291ba1e9fc03d9c433180794803440'
dest=PROJECT/'upstream'/'lejepa_identifiability_de7503f1';dest.mkdir(exist_ok=True,parents=True)
def acquire(row):
 url='https://raw.githubusercontent.com/klindtlab/lejepa-identifiability/'+selection['revision']+'/'+row['path']
 with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'world-model-source-audit'}),timeout=30) as response:raw=response.read()
 assert len(raw)==row['bytes'] and hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==row['git_blob_sha1']
 path=dest/row['path'];assert path.resolve().is_relative_to(dest.resolve());path.parent.mkdir(exist_ok=True,parents=True)
 if path.exists():assert path.read_bytes()==raw
 else:
  with path.open('xb') as f:f.write(raw)
 assert path.read_bytes()==raw
 return dict(**row,sha256=hashlib.sha256(raw).hexdigest(),server_path=str(path),url=url)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(acquire,selection['files']))
assert b'MIT License' in (dest/'LICENSE').read_bytes()
obj=dict(utc=utc(),status='AUTHOR_SOURCE_PINNED_NOT_EXECUTED',repository=selection['repository'],revision=selection['revision'],tree=selection['tree'],source_root=str(dest),files=rows,bytes=sum(r['bytes'] for r in rows),license='MIT',new_training_fits=0,new_performance_results=0,data_downloaded=False,shared_environment_modified=False,accepted_final_directions=0)
dump(OUT/'SOURCE_AUDIT71_LEJEPA_V1.json',obj);print(json.dumps({k:obj[k] for k in ['status','revision','bytes','new_training_fits','new_performance_results']}),flush=True)
