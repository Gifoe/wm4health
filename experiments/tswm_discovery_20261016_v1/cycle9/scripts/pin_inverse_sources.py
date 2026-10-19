"""Pin public small code/config only; no launcher execution, weights or datasets."""
import concurrent.futures,datetime,hashlib,json,pathlib,time
import requests
ROOT=pathlib.Path(__file__).resolve().parents[1]
REPOS={'ep_idm':'LeoToso/JEPA_control','acro':'manantomar/agent-centric-representations'}
def get(url):
 for trial in range(3):
  s=requests.Session();s.trust_env=False
  try:r=s.get(url,timeout=(5,15));r.raise_for_status();return r
  except requests.RequestException:
   if trial==2:raise
   time.sleep(.5)
def main():
 out=ROOT/'outputs';out.mkdir(parents=True,exist_ok=True)
 for key,repo in REPOS.items():
  folder=ROOT/'sources'/key;pinfile=folder/'SOURCE_PIN.json';treefile=out/(key+'_tree.json');old=json.loads(pinfile.read_text()) if pinfile.exists() else None
  if treefile.exists():tree=json.loads(treefile.read_text());rev=tree['sha']
  else:
   meta=get('https://api.github.com/repos/'+repo).json();rev=old['revision'] if old else get(f'https://api.github.com/repos/{repo}/commits/'+meta['default_branch']).json()['sha'];tree=get(f'https://api.github.com/repos/{repo}/git/trees/{rev}?recursive=1').json();treefile.write_text(json.dumps(tree,indent=2),encoding='utf-8')
  assert not tree.get('truncated',False)
  selected=[r for r in tree['tree'] if r['type']=='blob' and (pathlib.Path(r['path']).suffix in ['.py','.md','.yaml','.yml','.json','.toml','.txt'] or pathlib.Path(r['path']).name.startswith('LICENSE')) and r.get('size',0)<500000]
  assert sum(r.get('size',0) for r in selected)<25_000_000
  def fetch(row):
   rel=pathlib.PurePosixPath(row['path']);assert not rel.is_absolute() and '..' not in rel.parts;path=folder.joinpath(*rel.parts)
   if path.exists():raw=path.read_bytes()
   else:raw=get(f'https://raw.githubusercontent.com/{repo}/{rev}/'+row['path']).content
   assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==row['sha'],row['path']
   path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw);return {'path':row['path'],'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
  with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(fetch,selected))
  record={'repository':repo,'revision':rev,'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':rows,'scope':'Read-only small source/configuration snapshot; no datasets/checkpoints downloaded, no launcher executed'};pinfile.parent.mkdir(parents=True,exist_ok=True);pinfile.write_text(json.dumps(record,indent=2),encoding='utf-8');print(json.dumps({'key':key,'revision':rev,'files':len(rows),'bytes':sum(r['bytes'] for r in rows),'AGENTS':[r['path'] for r in rows if pathlib.Path(r['path']).name=='AGENTS.md']}),flush=True)
if __name__=='__main__':main()
