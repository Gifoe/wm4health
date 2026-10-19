"""Hash all cycle artifacts twice after scientific jobs and reports finish."""
import datetime,hashlib,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(1048576):h.update(b)
 return h.hexdigest()
def main():
 status=json.loads((ROOT/'outputs'/'task_status.json').read_text());assert status['cycle4']['scientific_jobs_complete'];assert not status['objective_achieved']
 files=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='ARTIFACT_MANIFEST.json' and not p.name.endswith(('.download','.partial','.tmp'))]
 rows=[{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(files)]
 report={'root_directory':'cycle4','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':rows,'count':len(rows),'bytes':sum(r['bytes'] for r in rows),'objective_achieved':False,'decision':'Evidence checkpoint only; no accepted research direction'}
 path=ROOT/'ARTIFACT_MANIFEST.json';path.write_text(json.dumps(report,indent=2),encoding='utf-8')
 for row in json.loads(path.read_text())['files']:
  p=ROOT/row['path'];assert p.stat().st_size==row['bytes'] and digest(p)==row['sha256'],row['path']
 print(json.dumps({'files':report['count'],'bytes':report['bytes'],'all_server_hashes_verified':True,'objective_achieved':False}))
if __name__=='__main__':main()
