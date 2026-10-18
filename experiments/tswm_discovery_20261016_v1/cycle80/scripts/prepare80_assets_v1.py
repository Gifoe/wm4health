"""Download only pinned base weights/config and one small public benchmark on server G."""
import sys,pathlib,urllib.request,time
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common80_v1 import *
def gitsha(p):
 b=p.read_bytes();return hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
def main():
 inv=read(PROJECT/'cycle79/outputs/PUBLIC_SOURCE_INVENTORY79_V1.json');model=inv['fim_ode'];dataset=inv['odebench'];assert model['revision']=='00875c253d025dd33fd70ae751089b1be3a7c174' and dataset['revision']=='91909865498f5d4bd69ea575527fee50ad329ef4'
 specs=[]
 for name in ['base_model/checkpoints/best-model/config.json','base_model/checkpoints/best-model/model.safetensors']:
  x=next(x for x in model['siblings'] if x['rfilename']==name);specs.append(dict(repo='FIM4Science/fim-ode',repo_type='model',revision=model['revision'],relative=name,path=storage('checkpoint')/'fim_ode_base_v1'/pathlib.Path(name).name,metadata=x))
 name='strogatz_extended.json';x=next(x for x in dataset['siblings'] if x['rfilename']==name);specs.append(dict(repo='FIM4Science/odebench',repo_type='dataset',revision=dataset['revision'],relative=name,path=storage('data')/'odebench_v1'/name,metadata=x))
 rows=[]
 for spec in specs:
  p=spec['path'];p.parent.mkdir(parents=True,exist_ok=True);x=spec['metadata'];urls=[];errors=[]
  if not p.exists():
   for host in ['https://hf-mirror.com','https://huggingface.co']:
    url=host+('/datasets/' if spec['repo_type']=='dataset' else '/')+spec['repo']+'/resolve/'+spec['revision']+'/'+spec['relative']+'?download=true';urls.append(url);part=p.with_name(p.name+('.mirror.part' if 'mirror' in host else '.official.part'))
    assert not part.exists(),('Preserve prior partial download, do not overwrite',str(part))
    try:
     with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'native-model-reproduction'}),timeout=50) as f,part.open('xb') as o:
      while chunk:=f.read(1048576):o.write(chunk)
     assert part.stat().st_size==x['size']
     if x.get('lfs'):assert sha(part)==x['lfs']['sha256']
     else:assert gitsha(part)==x['blobId']
     part.rename(p);break
    except Exception as exc:errors.append(dict(url=url,error=repr(exc)));print(json.dumps(dict(status='DOWNLOAD_ATTEMPT_FAILED_RETAINED',relative=spec['relative'],error=repr(exc))),flush=True)
   assert p.exists(),errors
  assert p.stat().st_size==x['size']
  if x.get('lfs'):assert sha(p)==x['lfs']['sha256']
  else:assert gitsha(p)==x['blobId']
  rows.append(dict(repo=spec['repo'],revision=spec['revision'],relative=spec['relative'],path=str(p),bytes=p.stat().st_size,sha256=sha(p),git_blob_sha1=x['blobId'],source_lfs=x.get('lfs'),urls_attempted=urls,failed_attempts=errors));print(json.dumps(dict(status='PINNED_ASSET_VERIFIED',relative=spec['relative'],bytes=p.stat().st_size,sha256=sha(p))),flush=True)
 dump(OUT/'ASSET_MANIFEST80_V1.json',dict(utc=utc(),assets=rows,weights_or_data_synchronized_locally=False,all_downloads_on_server_G=True,full_finetuned_or_training_corpora_downloaded=False))
if __name__=='__main__':main()
