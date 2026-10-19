"""Inspect a distinct native mechanism before choosing any new experiment."""
import base64,datetime,hashlib,json,pathlib
import requests
ROOT=pathlib.Path(__file__).resolve().parents[1]
REPO='cj7280/LENS'
def get(url):
 session=requests.Session();session.trust_env=False
 r=session.get(url,timeout=(5,20));r.raise_for_status();return r.json()
def main():
 out=ROOT/'outputs';out.mkdir(parents=True,exist_ok=True)
 treepath=out/'lens_tree.json'
 if treepath.exists(): tree=json.loads(treepath.read_text())
 else:
  metadata=get('https://api.github.com/repos/'+REPO)
  commit=get('https://api.github.com/repos/'+REPO+'/commits/'+metadata['default_branch'])
  (out/'lens_commit.json').write_text(json.dumps(commit,indent=2),encoding='utf-8')
  tree=get('https://api.github.com/repos/'+REPO+'/git/trees/'+commit['sha']+'?recursive=1')
  treepath.write_text(json.dumps(tree,indent=2),encoding='utf-8')
 assert not tree.get('truncated')
 blobs=[r for r in tree['tree'] if r['type']=='blob']
 agentrows=[r for r in blobs if pathlib.PurePosixPath(r['path']).name=='AGENTS.md']
 selected=agentrows+[r for r in blobs if r['path'] in ['README.md','training.py','utils.py','environment.yml'] or (r['path'].startswith(('src/','dataset_creation/')) and r['path'].endswith('.py'))]
 selected=list({r['path']:r for r in selected}.values());assert len(selected)<=35
 folder=ROOT/'sources/lens';rows=[]
 for row in selected:
  path=folder/row['path']
  if path.exists(): raw=path.read_bytes()
  else:
   blob=get(row['url']);assert blob['encoding']=='base64';raw=base64.b64decode(blob['content'])
   assert len(raw)<2000000
  assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==row['sha']
  path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
  rows.append({'path':row['path'],'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
  print('pinned',row['path'],len(raw),flush=True)
 pin={'repository':REPO,'revision':tree['sha'],'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Selected source audit only; no paper experiment reproduced, no model launcher executed, no movie/data/model download','AGENTS_files':[r['path'] for r in agentrows],'files':rows}
 (folder/'SOURCE_PIN.json').write_text(json.dumps(pin,indent=2),encoding='utf-8')
 print(json.dumps({'repository':REPO,'revision':tree['sha'],'tree_files':len(blobs),'AGENTS_files':pin['AGENTS_files'],'selected_files':len(rows)}))
if __name__=='__main__':main()
