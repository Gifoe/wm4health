"""Pin public native Info-LDM code; keep an explicit full-tree instruction audit."""
import base64,datetime,hashlib,json,pathlib
import requests
ROOT=pathlib.Path(__file__).resolve().parents[1];REPO='fmi-basel/identifiable-stochastic-nuisance'
def get(url):
 s=requests.Session();s.trust_env=False;r=s.get(url,timeout=(5,20));r.raise_for_status();return r.json()
def main():
 out=ROOT/'outputs';out.mkdir(exist_ok=True);folder=ROOT/'sources/infoldm';folder.mkdir(parents=True,exist_ok=True)
 tp=out/'infoldm_tree.json';cp=out/'infoldm_commit.json'
 if tp.exists():tree=json.loads(tp.read_text());commit=json.loads(cp.read_text())
 else:
  meta=get('https://api.github.com/repos/'+REPO);commit=get('https://api.github.com/repos/'+REPO+'/commits/'+meta['default_branch'])
  cp.write_text(json.dumps(commit,indent=2),encoding='utf-8');tree=get('https://api.github.com/repos/'+REPO+'/git/trees/'+commit['sha']+'?recursive=1');tp.write_text(json.dumps(tree,indent=2),encoding='utf-8')
 assert not tree.get('truncated');blobs=[r for r in tree['tree'] if r['type']=='blob']
 agents=[r for r in blobs if pathlib.PurePosixPath(r['path']).name=='AGENTS.md']
 selected=agents+[r for r in blobs if r['path'] in ['README.md','pyproject.toml','LICENSE','LICENSE.md','implicit_nuisance/__init__.py','implicit_nuisance/backbones/mlp.py','implicit_nuisance/backbones/__init__.py','implicit_nuisance/data/random_step_nuisance.py','implicit_nuisance/data/random_step.py','implicit_nuisance/utils/metrics.py'] or (r['path'].startswith(('implicit_nuisance/losses/','implicit_nuisance/methods/')) and r['path'].endswith('.py'))]
 selected=list({r['path']:r for r in selected}.values());assert len(selected)<=40,len(selected)
 rows=[]
 for row in selected:
  path=folder/row['path']
  if path.exists():raw=path.read_bytes()
  else:
   blob=get(row['url']);assert blob['encoding']=='base64';raw=base64.b64decode(blob['content'])
  assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==row['sha']
  path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
  rows.append({'path':row['path'],'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()});print('pinned',row['path'],len(raw),flush=True)
 pin={'repository':REPO,'revision':commit['sha'],'commit_tree':commit['commit']['tree']['sha'],'commit_date':commit['commit']['committer']['date'],'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Selected source review; no native launcher executed or full paper reproduction claimed','AGENTS_files':[r['path'] for r in agents],'files':rows}
 (folder/'SOURCE_PIN.json').write_text(json.dumps(pin,indent=2),encoding='utf-8');print(json.dumps({'revision':pin['revision'],'selected_files':len(rows),'AGENTS_files':pin['AGENTS_files']}))
if __name__=='__main__':main()
