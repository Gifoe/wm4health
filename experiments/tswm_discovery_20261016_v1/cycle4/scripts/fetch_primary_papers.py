"""Archive public primary papers locally; secure transport mirrors them to server."""
import concurrent.futures,datetime,hashlib,json,pathlib
import requests
ROOT=pathlib.Path(__file__).resolve().parents[1]
PAPERS=[('motion_jepa','2609.23881v1'),('spectral_alignment','2610.15429v1'),('action_jepa','2609.31161v1'),('switched_basis','2106.10888v3'),('variational_jepa','2601.14354v1'),('loss_conditioned_execution','2609.15801v1'),('phylatent','2608.05720v1'),('jepa_kmmvae','2610.13315v1'),('financial_world_modeling','2610.09048v1'),('heteroscedastic_volatility','2204.05806v1')]
def fetch(item):
 name,version=item;url='https://arxiv.org/pdf/'+version;folder=ROOT/'outputs'/'literature';folder.mkdir(parents=True,exist_ok=True)
 path=folder/(name+'_'+version+'.pdf')
 if not path.exists():
  session=requests.Session();session.trust_env=False
  r=session.get(url,timeout=75);r.raise_for_status();assert r.content.startswith(b'%PDF-')
  path.write_bytes(r.content)
 raw=path.read_bytes()
 return {'name':name,'arxiv_revision':version,'primary_url':url,'file':path.relative_to(ROOT).as_posix(),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
def main():
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(fetch,PAPERS))
 path=ROOT/'outputs'/'literature'/'registry.json';path.write_text(json.dumps(rows,indent=2),encoding='utf-8')
 print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
