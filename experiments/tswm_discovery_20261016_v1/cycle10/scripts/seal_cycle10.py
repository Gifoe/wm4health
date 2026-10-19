"""Immutable evidence manifest after verification, not research-goal completion."""
import datetime,hashlib,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(1048576):h.update(b)
 return h.hexdigest()
def main():
 assert not (ROOT/'ARTIFACT_MANIFEST.json').exists()
 verified=json.loads((ROOT/'outputs/evidence_verification.json').read_text())
 assert verified['all_saved_paths_and_summaries_verified'] and verified['rational_identities_verified']
 assert verified['native_checks']==15 and verified['covariance_fits']==120 and verified['randomized_controls']==1200
 status=json.loads((ROOT/'outputs/task_status.json').read_text());assert not status['objective_achieved']
 literature=json.loads((ROOT/'outputs/literature/registry.json').read_text());assert len(literature['papers'])==4 and not literature['errors']
 for row in literature['papers']:assert sha(ROOT/row['path'])==row['sha256']
 for name in ['literature_review.md','baseline_registry.md','experimental_results.md','negative_evidence.md','hypothesis_space.md','research_ideas.md','final_research_direction.md']:assert (ROOT/name).is_file()
 files=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='ARTIFACT_MANIFEST.json' and not p.name.endswith(('.partial','.download','.tmp'))]
 rows=[{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(files)]
 manifest={'root_directory':'cycle10','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':rows,'count':len(rows),'bytes':sum(r['bytes'] for r in rows),'objective_achieved':False,'decision':'retain exact model-class witness and existing repair; no final novel direction','exclusions':['Bytecode and incomplete transfers; full runtime outside cycle; relevant native source hashes checked against execution dependency; complete package retained in sealed cycle9']}
 (ROOT/'ARTIFACT_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
 for row in rows:assert sha(ROOT/row['path'])==row['sha256']
 print(json.dumps({'count':len(rows),'bytes':manifest['bytes'],'all_server_hashes_verified':True,'objective_achieved':False}))
if __name__=='__main__':main()
