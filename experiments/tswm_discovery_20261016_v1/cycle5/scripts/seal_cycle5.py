"""Seal reproducible research artifacts, excluding rebuildable numerical runtime."""
import datetime,hashlib,json,pathlib,subprocess,sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(1048576):h.update(b)
 return h.hexdigest()
def main():
 status=json.loads((ROOT/'outputs/task_status.json').read_text());assert status['cycle5']['scientific_jobs_complete'] and not status['objective_achieved']
 freeze=subprocess.run([sys.executable,'-m','pip','--python',str(ROOT/'.venv_certificate/Scripts/python.exe'),'freeze'],capture_output=True,text=True,check=True)
 (ROOT/'outputs/certificate_environment_freeze.txt').write_text(freeze.stdout,encoding='utf-8')
 files=[p for p in ROOT.rglob('*') if p.is_file() and '.venv_certificate' not in p.parts and '__pycache__' not in p.parts and p.name!='ARTIFACT_MANIFEST.json' and not p.name.endswith(('.download','.partial','.tmp'))]
 rows=[{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(files)]
 result={'root_directory':'cycle5','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':rows,'count':len(rows),'bytes':sum(r['bytes'] for r in rows),'exclusions':['Rebuildable .venv_certificate package runtime; pinned bootstrap and environment freeze included','Python bytecode and incomplete transfer files'],'objective_achieved':False,'decision':'Evidence checkpoint; four completed fits and18 controlled cases, no accepted research direction'}
 path=ROOT/'ARTIFACT_MANIFEST.json';path.write_text(json.dumps(result,indent=2),encoding='utf-8')
 for row in rows:
  p=ROOT/row['path'];assert p.stat().st_size==row['bytes'] and digest(p)==row['sha256'],row['path']
 print(json.dumps({'count':result['count'],'bytes':result['bytes'],'all_server_artifact_hashes_verified':True,'objective_achieved':False}))
if __name__=='__main__':main()
