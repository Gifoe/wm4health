"""Download public prerecorded numeric datasets directly on the research server."""
import datetime,hashlib,json,pathlib,time,zipfile
import requests
ROOT=pathlib.Path(__file__).resolve().parents[1]
ROWS=[('wt_walks_v1','19bb4e92cbe0b8dff49299b9b509ac36'),('wt_test_v1','5eed621f5b54e431c4442fc9cc947db6')]
def sha(path,algo='sha256'):
 h=hashlib.new(algo)
 with path.open('rb') as f:
  while data:=f.read(1048576):h.update(data)
 return h.hexdigest()
def main():
 session=requests.Session();session.trust_env=False;folder=ROOT/'data';folder.mkdir(exist_ok=True);rows=[]
 for name,md5 in ROWS:
  url='https://causalchamber.s3.eu-central-1.amazonaws.com/downloadables/'+name+'.zip';path=folder/(name+'.zip')
  staged=path.exists()
  if staged:size=path.stat().st_size
  else:
   response=session.head(url,timeout=(10,30));response.raise_for_status();size=int(response.headers.get('Content-Length',0))
  assert size<=512_000_000
  print(json.dumps({'dataset':name,'advertised_bytes':size}),flush=True)
  if not path.exists():
   temporary=path.with_suffix('.zip.partial');start=time.monotonic();last=start;total=0
   with session.get(url,stream=True,timeout=(15,45)) as response:
    response.raise_for_status()
    with temporary.open('wb') as f:
     for data in response.iter_content(262144):
      total+=len(data);assert total<=512_000_000;f.write(data)
      if time.monotonic()-last>20:print(json.dumps({'dataset':name,'received_bytes':total}),flush=True);last=time.monotonic()
   assert not size or temporary.stat().st_size==size
   assert sha(temporary,'md5')==md5,'Publisher MD5 mismatch'
   temporary.replace(path)
  assert sha(path,'md5')==md5
  entries=[]
  with zipfile.ZipFile(path) as z:
   for info in z.infolist():
    rel=pathlib.PurePosixPath(info.filename);assert not rel.is_absolute() and '..' not in rel.parts
    entries.append({'path':info.filename,'bytes':info.file_size})
    if not info.is_dir() and rel.suffix.lower() in ['.csv','.json','.yaml','.yml','.py','.md','.txt']:
     dst=folder/name/pathlib.Path(*rel.parts);assert dst.resolve().is_relative_to(folder.resolve());dst.parent.mkdir(parents=True,exist_ok=True);raw=z.read(info)
     if dst.exists():assert dst.read_bytes()==raw
     else:dst.write_bytes(raw)
  rows.append({'dataset':name,'url':url,'publisher_md5':md5,'bytes':path.stat().st_size,'sha256':sha(path),'entries':entries,'transport':'already staged and verified' if staged else 'direct HTTPS download','scope':'Public prerecorded data only; archive source scripts retained but not executed'})
  print(json.dumps({'dataset':name,'verified_bytes':path.stat().st_size,'archive_entries':len(entries)}),flush=True)
 out=ROOT/'outputs';out.mkdir(exist_ok=True);(out/'data_download.json').write_text(json.dumps({'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'datasets':rows},indent=2),encoding='utf-8')
if __name__=='__main__':main()
