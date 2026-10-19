"""Pin official EpiCF code and inspect data metadata before signal downloads."""
import datetime,hashlib,io,json,pathlib,shutil,subprocess,sys,zipfile
import requests
ROOT=pathlib.Path(__file__).resolve().parents[1]
def dump(path,value):path.parent.mkdir(exist_ok=True,parents=True);path.write_text(json.dumps(value,indent=2),encoding='utf-8')
def main():
 out=ROOT/'outputs';out.mkdir(exist_ok=True,parents=True);session=requests.Session();session.trust_env=False
 def get(url):
  try:r=session.get(url,timeout=45);r.raise_for_status();return r
  except requests.ConnectionError:
   if not url.startswith('https://zenodo.org/'):raise
   # Use the recorded public DNS answer, preserving Host/SNI/certificate checks.
   process=subprocess.run(['curl.exe','--fail','--location','--silent','--show-error','--connect-timeout','20','--max-time','90','--resolve','zenodo.org:443:188.185.40.132',url],capture_output=True,check=True)
   response=requests.Response();response.status_code=200;response._content=process.stdout;response.url=url;return response
 registry=[]
 for repo,scope in [('complex-ai-lab/epi-cf-benchmark','archive'),('complex-ai-lab/epi-diff-abm','metadata')]:
  info=get('https://api.github.com/repos/'+repo).json();commit=get('https://api.github.com/repos/'+repo+'/commits/'+info['default_branch']).json();sha=commit['sha']
  tree=get('https://api.github.com/repos/'+repo+'/git/trees/'+sha+'?recursive=1').json()
  name=repo.split('/')[-1];dst=ROOT/'sources'/name
  row={'repository':repo,'revision':sha,'commit_date':commit['commit']['committer']['date'],'license':info.get('license'),'scope':scope,'fetched_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
  dump(out/(name+'_tree.json'),tree)
  if scope=='archive':
   r=get('https://codeload.github.com/'+repo+'/zip/'+sha);row.update(archive_sha256=hashlib.sha256(r.content).hexdigest(),archive_bytes=len(r.content));dst.mkdir(exist_ok=True,parents=True)
   if (dst/'SOURCE_PIN.json').exists():assert json.loads((dst/'SOURCE_PIN.json').read_text())['revision']==sha,'Refuse to overwrite another source revision'
   with zipfile.ZipFile(io.BytesIO(r.content)) as archive:
    for item in archive.infolist():
     if item.is_dir():continue
     relative=pathlib.PurePosixPath(item.filename).parts[1:];assert relative and '..' not in relative
     path=dst.joinpath(*relative);assert path.resolve().is_relative_to(dst.resolve());path.parent.mkdir(exist_ok=True,parents=True);path.write_bytes(archive.read(item))
   dump(dst/'SOURCE_PIN.json',row)
  registry.append(row);print(json.dumps(row),flush=True)
 dump(out/'source_registry.json',registry)
 record=get('https://zenodo.org/api/records/20762823').json();dump(out/'zenodo_20762823_metadata.json',record)
 files=[{'key':f['key'],'size':f['size'],'checksum':f.get('checksum'),'links':f['links']} for f in record.get('files',[])]
 dump(out/'data_files.json',files);dump(out/'source_registry.json',registry)
 print(json.dumps({'data_files':files,'free_bytes':shutil.disk_usage(ROOT).free},indent=2),flush=True)
if __name__=='__main__':main()
