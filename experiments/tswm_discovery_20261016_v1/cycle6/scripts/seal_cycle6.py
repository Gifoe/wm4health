"""Hash a completed private evidence checkpoint, preserving all failed attempts."""
import datetime,hashlib,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  while data:=f.read(1048576):h.update(data)
 return h.hexdigest()
def main():
 status=json.loads((ROOT/'outputs/task_status.json').read_text());assert status['cycle6']['scientific_jobs_complete'] and not status['objective_achieved']
 assert json.loads((ROOT/'outputs/child_native_A_r1/process_exit.json').read_text())['exit_code']==0
 literature=json.loads((ROOT/'outputs/literature/registry.json').read_text());assert len(literature['papers'])==13 and not literature['errors']
 files=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='ARTIFACT_MANIFEST.json' and not p.name.endswith(('.download','.partial','.tmp'))]
 rows=[{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(files)]
 result={'root_directory':'cycle6','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':rows,'count':len(rows),'bytes':sum(r['bytes'] for r in rows),'exclusions':['Python bytecode and incomplete transfer files; numerical package runtime resides outside cycle6'],'objective_achieved':False,'decision':'Completed CHiLD source/config and controlled audits; no accepted research direction'}
 (ROOT/'ARTIFACT_MANIFEST.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
 for row in rows:
  path=ROOT/row['path'];assert path.stat().st_size==row['bytes'] and digest(path)==row['sha256'],row['path']
 print(json.dumps({'count':result['count'],'bytes':result['bytes'],'all_server_hashes_verified':True,'objective_achieved':False}))
if __name__=='__main__':main()
