"""Pin a small official source tree for a prospective loss-degeneracy audit."""
import datetime,hashlib,io,json,pathlib,time,zipfile,requests
ROOT=pathlib.Path(__file__).resolve().parents[1]
REPO='langotime/lenepa-milets-2026'
def main():
 s=requests.Session();s.trust_env=False
 def get(url):
  for attempt in range(3):
   try:response=s.get(url,timeout=45);response.raise_for_status();return response
   except requests.RequestException:
    if attempt==2:raise
    time.sleep(1)
 info=get('https://api.github.com/repos/'+REPO).json();commit=get('https://api.github.com/repos/'+REPO+'/commits/'+info['default_branch']).json();revision=commit['sha']
 tree=get(f'https://api.github.com/repos/{REPO}/git/trees/{revision}?recursive=1').json()
 assert sum(item.get('size',0) for item in tree['tree'])<100000000,'Inspect archive size before downloading'
 response=get(f'https://codeload.github.com/{REPO}/zip/{revision}')
 archive=zipfile.ZipFile(io.BytesIO(response.content))
 prefix=archive.infolist()[0].filename.split('/')[0]+'/'
 destination=ROOT/'sources'/'lenepa';destination.mkdir(exist_ok=True,parents=True)
 if (destination/'SOURCE_PIN.json').exists():assert json.loads((destination/'SOURCE_PIN.json').read_text())['revision']==revision
 rows=[]
 for item in tree['tree']:
  if item['type']!='blob' or pathlib.Path(item['path']).suffix not in {'.py','.yaml','.yml','.toml','.md','.tex','.json','.txt'} or item.get('size',0)>1500000:continue
  # Tables are unnecessary; preserve code, configs and the primary paper source.
  if item['path'].startswith(('results/','paper/figures/')):continue
  relative=pathlib.PurePosixPath(item['path']);assert '..' not in relative.parts and not relative.is_absolute()
  content=archive.read(prefix+item['path'])
  path=destination.joinpath(*relative.parts);path.parent.mkdir(exist_ok=True,parents=True);path.write_bytes(content)
  rows.append({'path':item['path'],'sha256':hashlib.sha256(content).hexdigest(),'bytes':len(content)})
 pin={'repository':REPO,'revision':revision,'commit_date':commit['commit']['committer']['date'],'license':info.get('license'),'fetched_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'code/config/paper-source only; no data, checkpoints or result tables','files':rows}
 (destination/'SOURCE_PIN.json').write_text(json.dumps(pin,indent=2),encoding='utf-8')
 print(json.dumps({'revision':revision,'files':len(rows),'bytes':sum(r['bytes'] for r in rows)}),flush=True)
if __name__=='__main__':main()
