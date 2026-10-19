"""Pin public ActSWM code without downloading unneeded signal data or weights."""
import base64,concurrent.futures,datetime,hashlib,json,pathlib,time
import requests
ROOT=pathlib.Path(__file__).resolve().parents[1]
def get(url):
 for trial in range(3):
  session=requests.Session();session.trust_env=False
  try:r=session.get(url,timeout=(8,18));r.raise_for_status();return r
  except requests.RequestException:
   if trial==2:raise
   time.sleep(1)
def main():
 repo='KeithBlackmore/ActSWM-';out=ROOT/'sources'/'actswm';out.mkdir(parents=True,exist_ok=True);pin=out/'SOURCE_PIN.json'
 old=json.loads(pin.read_text()) if pin.exists() else None;tree_path=ROOT/'outputs/actswm_tree.json'
 if tree_path.exists():tree=json.loads(tree_path.read_text());rev=tree['sha']
 else:
  metadata=get('https://api.github.com/repos/'+repo).json();rev=old['revision'] if old else get('https://api.github.com/repos/'+repo+'/commits/'+metadata['default_branch']).json()['sha']
  tree=get(f'https://api.github.com/repos/{repo}/git/trees/{rev}?recursive=1').json()
 assert not tree.get('truncated',False)
 (ROOT/'outputs/actswm_tree.json').write_text(json.dumps(tree,indent=2),encoding='utf-8')
 selected=[x for x in tree['tree'] if x['type']=='blob' and (pathlib.Path(x['path']).suffix in {'.py','.md','.txt','.json','.toml','.yaml','.yml','.sh'} or pathlib.Path(x['path']).name.startswith('LICENSE')) and x.get('size',0)<1_500_000]
 assert sum(x.get('size',0) for x in selected)<20_000_000
 def fetch(item):
  rel=pathlib.PurePosixPath(item['path']);assert not rel.is_absolute() and '..' not in rel.parts
  record=get(item['url']).json();assert record['encoding']=='base64';raw=base64.b64decode(record['content'])
  assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==item['sha']
  path=out.joinpath(*rel.parts);path.parent.mkdir(parents=True,exist_ok=True)
  if path.exists():assert path.read_bytes()==raw
  else:path.write_bytes(raw)
  return {'path':item['path'],'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:files=list(pool.map(fetch,selected))
 record=old or {'repository':repo,'revision':rev,'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Public small source/config only; no data archives/checkpoints','files':files}
 pin.write_text(json.dumps(record,indent=2),encoding='utf-8')
 print(json.dumps({'revision':rev,'files':len(files),'bytes':sum(x['bytes'] for x in files),'AGENTS':[x['path'] for x in files if pathlib.Path(x['path']).name=='AGENTS.md'],'large_assets':sorted([{'path':x['path'],'bytes':x.get('size',0)} for x in tree['tree'] if x['type']=='blob'],key=lambda a:a['bytes'],reverse=True)[:8]}),flush=True)
if __name__=='__main__':main()

