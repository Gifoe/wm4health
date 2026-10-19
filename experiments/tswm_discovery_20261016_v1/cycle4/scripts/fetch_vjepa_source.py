"""Inspect and pin only small author code/config files, never model/data assets."""
import concurrent.futures,datetime,hashlib,json,pathlib
import requests
ROOT=pathlib.Path(__file__).resolve().parents[1]
REPO='YongchaoHuang/VJEPA'
def request(url):
 s=requests.Session();s.trust_env=False;r=s.get(url,timeout=50);r.raise_for_status();return r
def main():
 info=request('https://api.github.com/repos/'+REPO).json()
 folder=ROOT/'sources'/'vjepa';folder.mkdir(exist_ok=True,parents=True);pinpath=folder/'SOURCE_PIN.json'
 if pinpath.exists():revision=json.loads(pinpath.read_text())['revision'];commit=request(f'https://api.github.com/repos/{REPO}/commits/{revision}').json()
 else:commit=request('https://api.github.com/repos/'+REPO+'/commits/'+info['default_branch']).json();revision=commit['sha']
 tree=request(f'https://api.github.com/repos/{REPO}/git/trees/{revision}?recursive=1').json();assert not tree.get('truncated',False)
 selected=[f for f in tree['tree'] if f['type']=='blob' and pathlib.Path(f['path']).suffix in {'.py','.md','.txt','.json','.toml','.yaml','.yml'} and f.get('size',0)<1500000 and not f['path'].startswith(('.git/',))]
 assert sum(f.get('size',0) for f in selected)<10000000
 def fetch(f):
  rel=pathlib.PurePosixPath(f['path']);assert '..' not in rel.parts and not rel.is_absolute()
  data=request(f'https://raw.githubusercontent.com/{REPO}/{revision}/{f["path"]}').content
  path=folder.joinpath(*rel.parts);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
  return {'path':f['path'],'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(fetch,selected))
 pin={'repository':REPO,'revision':revision,'commit_date':commit['commit']['committer']['date'],'fetched_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'small author code/config files only; no experiments executed yet','files':rows}
 pinpath.write_text(json.dumps(pin,indent=2),encoding='utf-8')
 print(json.dumps({'repository':REPO,'revision':revision,'files':len(rows),'bytes':sum(r['bytes'] for r in rows),'agent_files':[r['path'] for r in rows if pathlib.Path(r['path']).name=='AGENTS.md']}))
if __name__=='__main__':main()
