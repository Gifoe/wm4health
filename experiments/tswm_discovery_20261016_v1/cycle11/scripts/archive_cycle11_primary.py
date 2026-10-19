"""Small primary archive for the score audit and next-source triage."""
import hashlib,json,pathlib,shutil
import requests
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/literature';OUT.mkdir(parents=True,exist_ok=True)
PAPERS=[('info_ldm','2609.37789v1','https://arxiv.org/pdf/2609.37789v1'),('self_predictive_rl','2401.08898v3','https://arxiv.org/pdf/2401.08898v3'),('log_loss','1403.4679v2','https://arxiv.org/pdf/1403.4679v2'),('cpc_anchor','ICLR2026 published','https://proceedings.iclr.cc/paper_files/paper/2026/file/304e0ae755d5d7e9016e3d05700c35b2-Paper-Conference.pdf'),('qg_closure','2603.28671v1','https://arxiv.org/pdf/2603.28671v1')]
def main():
 rows=[];errors=[];s=requests.Session();s.trust_env=False
 for key,version,url in PAPERS:
  path=OUT/(key+'.pdf')
  try:
   if not path.exists():
    r=s.get(url,timeout=(5,25));r.raise_for_status();assert r.content.startswith(b'%PDF');path.write_bytes(r.content)
   data=path.read_bytes();assert data.startswith(b'%PDF');rows.append({'key':key,'version':version,'url':url,'path':path.relative_to(ROOT).as_posix(),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()});print(key,len(data),flush=True)
  except (requests.RequestException,OSError,AssertionError) as e:errors.append({'key':key,'url':url,'error':repr(e)});print('failed',key,type(e).__name__,flush=True)
 for key,name,version,url in [('branch_jepa','branch_jepa_2607.05238v3.pdf','2607.05238v3','https://arxiv.org/pdf/2607.05238v3')]:
  original=ROOT.parent/'cycle6/outputs/literature'/name;assert original.is_file();path=OUT/(key+'.pdf');shutil.copyfile(original,path);data=path.read_bytes();rows.append({'key':key,'version':version,'url':url,'path':path.relative_to(ROOT).as_posix(),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'copy_of_verified_prior_archive':True})
 (OUT/'registry.json').write_text(json.dumps({'papers':rows,'errors':errors,'scope':'Primary versions; archived source does not imply full proof audit or method reproduction.'},indent=2),encoding='utf-8')
if __name__=='__main__':main()
