"""Download the official bounded archive directly on the private server."""
import datetime,hashlib,json,pathlib,shutil
import requests
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs';DATA=ROOT/'data';DATA.mkdir(parents=True,exist_ok=True)
def hashes(p):
 md5=hashlib.md5();sha=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(1048576):md5.update(b);sha.update(b)
 return md5.hexdigest(),sha.hexdigest()
def main():
 record=json.loads((OUT/'qg_checkpoint_record.json').read_text());row=next(r for r in record['files'] if r['key']=='models_archive.tar.gz');size=row['size'];assert size==316038179 and row['checksum']=='md5:db83828a75e102f41611ee9032f5885d'
 assert shutil.disk_usage(ROOT).free>2_000_000_000
 dest=DATA/'models_archive.tar.gz';partial=dest.with_suffix(dest.suffix+'.partial')
 if not dest.exists():
  offset=partial.stat().st_size if partial.exists() else 0;assert offset<=size
  if offset<size:
   s=requests.Session();s.trust_env=False;headers={'Range':f'bytes={offset}-'} if offset else {}
   with s.get(row['links']['self'],headers=headers,stream=True,timeout=(10,30)) as response:
    response.raise_for_status()
    if offset:assert response.status_code==206 and response.headers.get('Content-Range','').startswith(f'bytes {offset}-')
    with partial.open('ab' if offset else 'wb') as f:
     last=offset
     for chunk in response.iter_content(1048576):
      f.write(chunk);offset+=len(chunk);assert offset<=size
      if offset-last>=32*1048576:print('downloaded_bytes',offset,'of',size,flush=True);last=offset
  assert partial.stat().st_size==size;md5,sha=hashes(partial);assert 'md5:'+md5==row['checksum'];partial.replace(dest)
 assert dest.stat().st_size==size;md5,sha=hashes(dest);assert 'md5:'+md5==row['checksum']
 result={'file':'data/models_archive.tar.gz','bytes':size,'official_record':'https://zenodo.org/records/19335706','official_checksum':row['checksum'],'sha256':sha,'downloaded_directly_on_server':True,'verified':True,'extracted':False,'deserialized':False,'scientific_inference_executed':False,'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 (OUT/'checkpoint_download.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
