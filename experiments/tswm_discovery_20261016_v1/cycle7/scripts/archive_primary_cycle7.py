"""Small primary-document archive with pinned versions and bounded transfer."""
import concurrent.futures,datetime,hashlib,json,pathlib
import requests
ROOT=pathlib.Path(__file__).resolve().parents[1]
SOURCES=[('dwm','https://arxiv.org/pdf/2607.18715v1'),('intervention_gap','https://arxiv.org/pdf/2608.29998v1'),('actswm','https://arxiv.org/pdf/2607.26712v1'),('mirabench','https://arxiv.org/pdf/2605.29360v1'),('slim','https://arxiv.org/pdf/2610.03137v1'),('caer','https://arxiv.org/pdf/2608.30897v1'),('flex_wam','https://arxiv.org/pdf/2610.05483v1'),('counterfactual_action_eval','https://arxiv.org/pdf/2610.02860v1'),('sensorimotor','https://arxiv.org/pdf/2606.20104v1'),('csa_pfn','https://arxiv.org/pdf/2605.10590v1'),('dotime','https://arxiv.org/pdf/2607.27263v1'),('counterfactual_debugging','https://mingxuan.me/assets/papers/ctf_debugging.pdf'),('causal_longpfn','https://arxiv.org/pdf/2606.05797v1'),('action_homomorphism_ijcai2003','https://www.ijcai.org/Proceedings/03/Papers/145.pdf')]
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
