"""Pin a published stochastic physical-world-model baseline before new tests."""
import datetime,hashlib,io,json,pathlib,re,zipfile
import requests
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs';OUT.mkdir(parents=True,exist_ok=True)
REPO='mtbrolly/qg_closure'
def request(url,json_result=True):
 s=requests.Session();s.trust_env=False;r=s.get(url,timeout=(5,25));r.raise_for_status();return r.json() if json_result else r.content
def main():
 cp=OUT/'qg_commit.json';tp=OUT/'qg_tree.json'
 if cp.exists():commit=json.loads(cp.read_text());tree=json.loads(tp.read_text())
 else:
  commit=request('https://api.github.com/repos/'+REPO+'/commits/HEAD');cp.write_text(json.dumps(commit,indent=2),encoding='utf-8')
  tree=request('https://api.github.com/repos/'+REPO+'/git/trees/'+commit['sha']+'?recursive=1');tp.write_text(json.dumps(tree,indent=2),encoding='utf-8')
 assert not tree.get('truncated');entries={r['path']:r for r in tree['tree'] if r['type']=='blob'}
 assert sum(r.get('size',0) for r in entries.values())<32000000,'Review large repository contents first'
 archive=OUT/'qg_source_archive.zip'
 if not archive.exists():
  raw=request('https://codeload.github.com/'+REPO+'/zip/'+commit['sha'],False);assert len(raw)<32000000;archive.write_bytes(raw)
 folder=ROOT/'sources/qg_closure';folder.mkdir(parents=True,exist_ok=True);rows=[]
 with zipfile.ZipFile(archive) as z:
  for member in z.infolist():
   if member.is_dir():continue
   parts=pathlib.PurePosixPath(member.filename).parts;rel=pathlib.PurePosixPath(*parts[1:]);assert '..' not in rel.parts and not rel.is_absolute()
   name=rel.as_posix();assert name in entries,name;data=z.read(member)
   assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==entries[name]['sha'],name
   path=folder.joinpath(*rel.parts);assert path.resolve().is_relative_to(folder.resolve());path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
   rows.append({'path':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
 assert len(rows)==len(entries),(len(rows),len(entries))
 flags=[]
 for row in rows:
  if pathlib.PurePosixPath(row['path']).suffix in ['.py','.sh','.yaml','.yml']:
   raw=(folder/row['path']).read_bytes()
   if re.search(rb'(?:WANDB_API_KEY\s*[=:]\s*[\x22\x27][A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{24,})',raw):flags.append(row['path'])
 pin={'repository':REPO,'revision':commit['sha'],'commit_tree':commit['commit']['tree']['sha'],'commit_date':commit['commit']['committer']['date'],'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'scope':'Complete source snapshot only; no pipeline, dependency installation, model/data download or scientific evaluation yet','AGENTS_files':[name for name in entries if pathlib.PurePosixPath(name).name=='AGENTS.md'],'credential_pattern_flags_filenames_only':flags,'files':rows}
 (folder/'SOURCE_PIN.json').write_text(json.dumps(pin,indent=2),encoding='utf-8');print(json.dumps({k:pin[k] for k in ['repository','revision','commit_date','AGENTS_files','credential_pattern_flags_filenames_only']}));print('verified complete source files',len(rows))
 zenodo=request('https://zenodo.org/api/records/19335706');(OUT/'qg_checkpoint_record.json').write_text(json.dumps(zenodo,indent=2),encoding='utf-8');print(json.dumps({'checkpoint_files':[{'key':r['key'],'size':r['size'],'checksum':r['checksum']} for r in zenodo['files']]}))
 for item in zenodo['files']:
  if item['key']=='models_archive_description.md':
   raw=request(item['links']['self'],False);assert len(raw)<10000;assert 'md5:'+hashlib.md5(raw).hexdigest()==item['checksum'];(OUT/'models_archive_description.md').write_bytes(raw);print(raw.decode('utf-8'))
if __name__=='__main__':main()
