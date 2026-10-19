"""Small immutable primary-paper archive, no secondary summaries."""
import hashlib,json,pathlib,time
import requests
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/literature';OUT.mkdir(parents=True,exist_ok=True)
PAPERS={'ac_mtm':('2608.17542v1','https://arxiv.org/pdf/2608.17542v1'),'acro':('2211.00164v2','https://arxiv.org/pdf/2211.00164v2'),'action_bisimulation':('2403.16369v2','https://arxiv.org/pdf/2403.16369v2'),'response_probe':('2610.06798v1','https://arxiv.org/pdf/2610.06798v1'),'conditional_kernel_nce':('1906.02236v2','https://arxiv.org/pdf/1906.02236v2')}
def main():
 rows=[];errors=[]
 for key,(version,url) in PAPERS.items():
  path=OUT/(key+'.pdf')
  try:
   if not path.exists():
    session=requests.Session();session.trust_env=False;r=session.get(url,timeout=(5,20));r.raise_for_status();assert r.content.startswith(b'%PDF');path.write_bytes(r.content)
   raw=path.read_bytes();assert raw.startswith(b'%PDF');rows.append({'key':key,'version':version,'url':url,'path':path.relative_to(ROOT).as_posix(),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()});print(key,len(raw),flush=True)
  except (requests.RequestException,OSError) as error:errors.append({'key':key,'url':url,'error':repr(error)});print('failed',key,type(error).__name__,flush=True)
 (OUT/'registry.json').write_text(json.dumps({'papers':rows,'errors':errors,'scope':'Primary preprint PDFs; no venue acceptance inferred.'},indent=2),encoding='utf-8')
if __name__=='__main__':main()
