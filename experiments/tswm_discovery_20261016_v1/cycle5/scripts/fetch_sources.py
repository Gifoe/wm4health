"""Pin inspected small source files; do not fetch datasets/checkpoints/assets."""
import concurrent.futures,datetime,hashlib,json,pathlib,time,urllib.parse
import requests
ROOT=pathlib.Path(__file__).resolve().parents[1]
SOURCES=[('cofe','polars8948/COFE',None),('kuma','SuperBearcafedo/Review','KUMA-Official')]
def get(url):
 for attempt in range(3):
  s=requests.Session();s.trust_env=False
  try:r=s.get(url,timeout=50);r.raise_for_status();return r
  except requests.RequestException:
   if attempt==2:raise
   time.sleep(1)
def main():
 registry=[]
 for name,repo,branch in SOURCES:
  folder=ROOT/'sources'/name;folder.mkdir(parents=True,exist_ok=True);pinpath=folder/'SOURCE_PIN.json'
  info=get('https://api.github.com/repos/'+repo).json()
  old=json.loads(pinpath.read_text()) if pinpath.exists() else None
  revision=old['revision'] if old else get(f'https://api.github.com/repos/{repo}/commits/'+urllib.parse.quote(branch or info['default_branch'],safe='')).json()['sha']
  tree=get(f'https://api.github.com/repos/{repo}/git/trees/{revision}?recursive=1').json();assert not tree.get('truncated',False)
  assets=[{'path':x['path'],'bytes':x.get('size',0),'type':x['type']} for x in tree['tree'] if x['type'] in ['blob','commit']]
  selected=[x for x in tree['tree'] if x['type']=='blob' and (pathlib.Path(x['path']).suffix in {'.py','.md','.yaml','.yml','.toml','.txt','.json','.sh'} or pathlib.Path(x['path']).name=='LICENSE') and x.get('size',0)<=1500000 and not x['path'].startswith(('figs/','results/','.github/'))]
  assert sum(x.get('size',0) for x in selected)<15000000
  def fetch(item):
   rel=pathlib.PurePosixPath(item['path']);assert not rel.is_absolute() and '..' not in rel.parts
   content=get(f'https://raw.githubusercontent.com/{repo}/{revision}/{item["path"]}').content
   target=folder.joinpath(*rel.parts);target.parent.mkdir(parents=True,exist_ok=True)
   if target.exists():assert target.read_bytes()==content
   else:target.write_bytes(content)
   return {'path':item['path'],'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()}
  with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:files=list(pool.map(fetch,selected))
  pin=old or {'repository':repo,'branch':branch or info['default_branch'],'revision':revision,'fetched_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'license':info.get('license'),'scope':'small author code/config/docs only; no data or weights','files':files}
  pinpath.write_text(json.dumps(pin,indent=2),encoding='utf-8');registry.append(pin)
  dump=ROOT/'outputs'/(name+'_tree.json');dump.write_text(json.dumps(assets,indent=2),encoding='utf-8')
  print(json.dumps({'name':name,'repository':repo,'revision':revision,'source_files':len(files),'source_bytes':sum(x['bytes'] for x in files),'agent_files':[x['path'] for x in files if pathlib.Path(x['path']).name=='AGENTS.md'],'asset_largest':sorted(assets,key=lambda x:x['bytes'],reverse=True)[:5]}),flush=True)
 (ROOT/'outputs'/'source_registry.json').write_text(json.dumps(registry,indent=2),encoding='utf-8')
if __name__=='__main__':main()
