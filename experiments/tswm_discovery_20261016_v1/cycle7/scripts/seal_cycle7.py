"""Seal a completed private evidence checkpoint without declaring goal success."""
import datetime,hashlib,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  while data:=f.read(1048576):h.update(data)
 return h.hexdigest()
def main():
 assert not (ROOT/'ARTIFACT_MANIFEST.json').exists(),'Sealed evidence must not be overwritten'
 status=json.loads((ROOT/'outputs/task_status.json').read_text());assert status['cycle7']['scientific_jobs_complete'] and not status['objective_achieved']
 assert len(json.loads((ROOT/'outputs/action_equivalence_pilot/results.json').read_text()))==27
 assert len(json.loads((ROOT/'outputs/objective_certificate/result.json').read_text())['rows'])==6
 assert len(json.loads((ROOT/'outputs/literature/registry.json').read_text())['papers'])==14
 exits=json.loads((ROOT/'outputs/process_exits.json').read_text());assert exits['native_action_probe']==0 and exits['oracle_objective_certificate']==0
 names=['literature_review.md','baseline_registry.md','experimental_results.md','negative_evidence.md','hypothesis_space.md','research_ideas.md','final_research_direction.md']
 assert all((ROOT/n).is_file() for n in names)
 files=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='ARTIFACT_MANIFEST.json' and not p.name.endswith(('.download','.partial','.tmp'))]
 rows=[{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(files)]
 result={'root_directory':'cycle7','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':rows,'count':len(rows),'bytes':sum(r['bytes'] for r in rows),'exclusions':['Python bytecode and incomplete transfers; external runtime not copied'],'objective_achieved':False,'decision':'Objective mismatch supported locally; no new method need or accepted final direction'}
 (ROOT/'ARTIFACT_MANIFEST.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
 for r in rows:assert digest(ROOT/r['path'])==r['sha256']
 print(json.dumps({'count':len(rows),'bytes':result['bytes'],'all_server_hashes_verified':True,'objective_achieved':False}))
if __name__=='__main__':main()
