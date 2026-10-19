"""Archive next-scope code availability without downloading benchmark data."""
import concurrent.futures,datetime,hashlib,json,pathlib,urllib.parse
from fetch_sources import get,ROOT
def main():
 registry=[]
 for name,repo in [('metailad','GoForIt-YYB/Meta-iLaD'),('timebench','zqiao11/TIME')]:
  out=ROOT/'sources'/name;out.mkdir(parents=True,exist_ok=True);pin=out/'SOURCE_PIN.json'
  metadata=get('https://api.github.com/repos/'+repo).json();old=json.loads(pin.read_text()) if pin.exists() else None
  revision=old['revision'] if old else get('https://api.github.com/repos/'+repo+'/commits/'+metadata['default_branch']).json()['sha']
  tree=get(f'https://api.github.com/repos/{repo}/git/trees/{revision}?recursive=1').json();assert not tree.get('truncated',False)
  selected=[x for x in tree['tree'] if x['type']=='blob' and (pathlib.Path(x['path']).suffix in {'.py','.md','.toml','.json','.yaml','.yml','.txt','.sh'} or pathlib.Path(x['path']).name=='LICENSE') and x.get('size',0)<1_000_000]
  assert sum(x.get('size',0) for x in selected)<10_000_000
  def fetch(item):
   raw=get(f'https://raw.githubusercontent.com/{repo}/{revision}/'+item['path']).content;path=out/item['path'];path.parent.mkdir(parents=True,exist_ok=True)
   if path.exists():assert path.read_bytes()==raw
   else:path.write_bytes(raw)
   return {'path':item['path'],'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
  with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:files=list(pool.map(fetch,selected))
  result=old or {'repository':repo,'revision':revision,'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':files,'scope':'Small public code/config/documentation only, no datasets or checkpoints','code_files':sum(x['path'].endswith('.py') for x in files),'AGENTS':[x['path'] for x in files if pathlib.Path(x['path']).name=='AGENTS.md']}
  pin.write_text(json.dumps(result,indent=2),encoding='utf-8');registry.append(result);print(json.dumps({k:v for k,v in result.items() if k!='files'}),flush=True)
 (ROOT/'outputs'/'followup_source_registry.json').write_text(json.dumps(registry,indent=2),encoding='utf-8')
if __name__=='__main__':main()
