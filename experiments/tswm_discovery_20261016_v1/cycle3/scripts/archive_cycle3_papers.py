"""Archive public primary PDFs and exact public paper metadata, without training."""
import datetime,hashlib,json,pathlib,requests
ROOT=pathlib.Path(__file__).resolve().parents[1]
PAPERS=[('epicf','2606.05692v1'),('prodyn','2507.15774v2'),('semigroup_jepa','2609.10464v1'),('one_step_latent','2609.36227v1'),('action_geometry','2610.16644v1'),('control_predictability','2607.10362v1'),('lenepa','2607.00958v1'),('motion_jepa','2609.23881v1'),('tc_lewm','2607.26924v1')]
def main():
 s=requests.Session();s.trust_env=False
 out=ROOT/'outputs'/'literature';out.mkdir(exist_ok=True,parents=True);rows=[]
 for name,revision in PAPERS:
  path=out/f'{name}_{revision}.pdf';url='https://arxiv.org/pdf/'+revision
  if not path.exists():
   response=s.get(url,timeout=60);response.raise_for_status();assert response.content.startswith(b'%PDF-')
   path.write_bytes(response.content)
  rows.append({'key':name,'arxiv_revision':revision,'url':url,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size,'downloaded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
  print(name,path.stat().st_size,flush=True)
 (out/'paper_registry.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
if __name__=='__main__':main()
