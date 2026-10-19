"""Download only two required public files, validating Git content hashes."""
import datetime,hashlib,json,pathlib,sys
import requests
ROOT=pathlib.Path(__file__).resolve().parents[1]
def get(url,timeout=25):
 s=requests.Session();s.trust_env=False;r=s.get(url,timeout=timeout);r.raise_for_status();return r
def main():
 planpath=ROOT/'data'/'download_plan.json'
 if '--prepare' in sys.argv:
  rows=[]
  for repo,ref,name,dest in [('SuperBearcafedo/Review','8ed21081366309e313f821f3dc760f6ac945ac4b','dataset/PEMS/PEMS03.npz','PEMS03.npz'),('zhouhaoyi/ETDataset','main','ETT-small/ETTh1.csv','ETTh1.csv')]:
   if ref=='main':ref=get(f'https://api.github.com/repos/{repo}/commits/{ref}').json()['sha']
   meta=get(f'https://api.github.com/repos/{repo}/contents/{name}?ref={ref}').json()
   rows.append({'repository':repo,'revision':ref,'source_path':name,'destination':dest,'bytes':meta['size'],'git_blob_sha1':meta['sha'],'url':f'https://raw.githubusercontent.com/{repo}/{ref}/{name}'})
  planpath.write_text(json.dumps(rows,indent=2),encoding='utf-8');print(json.dumps(rows));return
 rows=json.loads(planpath.read_text());out=[]
 for row in rows:
  p=ROOT/'data'/row['destination']
  if not p.exists():
   data=get(row['url']).content
   assert len(data)==row['bytes'];assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==row['git_blob_sha1']
   p.write_bytes(data)
  data=p.read_bytes();assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==row['git_blob_sha1']
  out.append({**row,'sha256':hashlib.sha256(data).hexdigest(),'verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()});print('Verified',row['destination'],len(data),flush=True)
 (ROOT/'outputs'/'dataset_registry.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
if __name__=='__main__':main()
