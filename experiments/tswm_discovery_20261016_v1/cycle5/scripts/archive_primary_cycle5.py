"""Fetch only small primary papers and HTML, locally; transport verifies copies."""
import concurrent.futures,datetime,hashlib,json,pathlib
import requests
ROOT=pathlib.Path(__file__).resolve().parents[1]
SOURCES=[
 ('cofe_ijcai2026','https://www.ijcai.org/proceedings/2026/0602.pdf','pdf'),
 ('kuma_icml2026','https://raw.githubusercontent.com/mlresearch/v306/main/assets/xiong26j/xiong26j.pdf','pdf'),
 ('haugh_singal_icml2023','https://proceedings.mlr.press/v202/haugh23a/haugh23a.pdf','pdf'),
 ('worldkernel_2606.10934v1','https://arxiv.org/pdf/2606.10934v1','pdf'),
 ('ivdfm_2603.22886v1','https://arxiv.org/pdf/2603.22886v1','pdf'),
 ('physical_identifiability_2607.27017v1','https://arxiv.org/pdf/2607.27017v1','pdf'),
 ('counterfactual_flow_2610.01193v1','https://arxiv.org/pdf/2610.01193v1','pdf'),
 ('dr_kalman_neurips2018','https://proceedings.neurips.cc/paper/2018/file/15212f24321aa2c3dc8e9acf820f3c15-Paper.pdf','pdf'),
 ('orthogonal_sdp_2110.05701v1','https://arxiv.org/pdf/2110.05701v1','pdf'),
 ('robust_pnscm_2608.02893v1','https://arxiv.org/pdf/2608.02893v1','pdf'),
 ('ossm_2609.33566v1','https://arxiv.org/pdf/2609.33566v1','pdf'),
 ('pythia_2610.05240v1','https://arxiv.org/pdf/2610.05240v1','pdf'),
]
def fetch(row):
 name,url,ext=row;folder=ROOT/'outputs'/'literature';folder.mkdir(parents=True,exist_ok=True);path=folder/(name+'.'+ext)
 if not path.exists():
  session=requests.Session();session.trust_env=False
  response=session.get(url,timeout=90);response.raise_for_status();raw=response.content
  assert len(raw)<12_000_000 and raw.startswith(b'%PDF-'),url
  path.write_bytes(raw)
 raw=path.read_bytes()
 return {'name':name,'url':url,'path':path.relative_to(ROOT).as_posix(),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
def main():
 rows=[];errors=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  futures={pool.submit(fetch,row):row for row in SOURCES}
  for f,row in futures.items():
   try:rows.append(f.result())
   except Exception as error:errors.append({'name':row[0],'url':row[1],'error':repr(error)})
 result={'papers':rows,'errors':errors};(ROOT/'outputs'/'literature'/'registry.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
