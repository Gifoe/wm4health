"""Retain bounded primary sources; never interpret a failed download as a claim."""
import datetime,hashlib,json,pathlib
import requests
ROOT=pathlib.Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs/literature';OUT.mkdir(parents=True,exist_ok=True)
PAPERS=[
 ('temporal_iv','JMLR25(302),2024','https://jmlr.org/papers/volume25/22-0262/22-0262.pdf'),
 ('hidden_var','ICML/PMLR37,2015','https://proceedings.mlr.press/v37/geiger15.pdf'),
 ('cafe','2609.32167v1','https://arxiv.org/pdf/2609.32167v1'),
 ('psg_jepa','2608.06799v1','https://arxiv.org/pdf/2608.06799v1')]
def main():
 rows=[];errors=[];session=requests.Session();session.trust_env=False
 for key,version,url in PAPERS:
  path=OUT/(key+'.pdf')
  try:
   if not path.exists():
    r=session.get(url,timeout=(5,25));r.raise_for_status();assert r.content.startswith(b'%PDF');path.write_bytes(r.content)
   data=path.read_bytes();assert data.startswith(b'%PDF')
   rows.append({'key':key,'version':version,'url':url,'path':path.relative_to(ROOT).as_posix(),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()});print(key,len(data),flush=True)
  except (requests.RequestException,OSError,AssertionError) as e:
   errors.append({'key':key,'url':url,'error':repr(e)});print('archive failed',key,type(e).__name__,flush=True)
 (OUT/'registry.json').write_text(json.dumps({'papers':rows,'errors':errors,'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Primary PDFs retained privately; preprint versions do not imply acceptance.'},indent=2),encoding='utf-8')
if __name__=='__main__':main()
