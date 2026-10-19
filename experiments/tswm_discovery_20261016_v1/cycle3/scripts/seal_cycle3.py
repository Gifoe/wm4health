"""Write and independently re-read the server artifact manifest after completion."""
import datetime,hashlib,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  while data:=f.read(1024*1024):h.update(data)
 return h.hexdigest()
def main():
 assert json.loads((ROOT/'outputs'/'task_status.json').read_text())['cycle3']['scientific_jobs_complete']
 paths=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='ARTIFACT_MANIFEST.json' and not p.name.endswith(('.partial','.download'))]
 rows=[{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(paths)]
 manifest={'root_directory':'cycle3','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'count':len(rows),'bytes':sum(r['bytes'] for r in rows),'files':rows,'objective_achieved':False,'decision':'Evidence checkpoint only; overall research discovery continues'}
 path=ROOT/'ARTIFACT_MANIFEST.json';path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
 for row in json.loads(path.read_text())['files']:
  artifact=ROOT/row['path'];assert artifact.stat().st_size==row['bytes'] and digest(artifact)==row['sha256'],row['path']
 print(json.dumps({'files':len(rows),'bytes':manifest['bytes'],'all_hashes_verified':True,'objective_achieved':False}))
if __name__=='__main__':main()
