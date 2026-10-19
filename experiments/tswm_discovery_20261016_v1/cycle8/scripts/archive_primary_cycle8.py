"""Small primary-document archive with pinned versions and bounded transfer."""
import concurrent.futures,datetime,hashlib,json,pathlib
import requests
ROOT=pathlib.Path(__file__).resolve().parents[1]
SOURCES=[('causal_chambers','https://arxiv.org/pdf/2404.11341v2'),('tawm','https://arxiv.org/pdf/2506.08441v1'),('endoclock','https://arxiv.org/pdf/2608.09266v1'),('hypernsde','https://arxiv.org/pdf/2610.07383v1'),('sensing_reliability','https://arxiv.org/pdf/2609.07126v1'),('cider_fm','https://arxiv.org/pdf/2609.39523v1'),('tesar_cde','https://arxiv.org/pdf/2306.04255v1'),('te_cde','https://arxiv.org/pdf/2206.08311v1'),('cf_jepa','https://arxiv.org/pdf/2610.00727v1'),('ep_idm','https://arxiv.org/pdf/2610.07540v1')]
def fetch(row):
 name,url=row;folder=ROOT/'outputs/literature';folder.mkdir(parents=True,exist_ok=True);path=folder/(name+'.pdf')
 if not path.exists():
  session=requests.Session();session.trust_env=False;parts=[];size=0
  with session.get(url,timeout=(10,35),stream=True) as response:
   response.raise_for_status()
   for chunk in response.iter_content(131072):
    size+=len(chunk)
    if size>24_000_000:raise ValueError('PDF cap exceeded; retain verified primary HTML reading instead')
    parts.append(chunk)
  raw=b''.join(parts);assert raw.startswith(b'%PDF-');path.write_bytes(raw)
 raw=path.read_bytes();return {'name':name,'url':url,'path':path.relative_to(ROOT).as_posix(),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
def main():
 rows=[];errors=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  jobs={pool.submit(fetch,row):row for row in SOURCES}
  for future,row in jobs.items():
   try:rows.append(future.result());print('Archived',row[0],flush=True)
   except Exception as error:errors.append({'name':row[0],'url':row[1],'error':repr(error)});print('Unavailable',row[0],type(error).__name__,flush=True)
 (ROOT/'outputs/literature/registry.json').write_text(json.dumps({'papers':rows,'errors':errors},indent=2),encoding='utf-8');print(json.dumps({'archived':len(rows),'unavailable':len(errors)}))
if __name__=='__main__':main()
