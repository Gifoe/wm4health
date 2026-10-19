"""Seal verified private evidence, with full research goal still active."""
import datetime,hashlib,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  while b:=f.read(1048576):h.update(b)
 return h.hexdigest()
def main():
 assert not (ROOT/'ARTIFACT_MANIFEST.json').exists()
 status=json.loads((ROOT/'outputs/task_status.json').read_text(encoding='utf-8'));assert status['cycle9']['scientific_jobs_complete'] and not status['objective_achieved']
 verification=json.loads((ROOT/'outputs/evidence_verification.json').read_text(encoding='utf-8'));assert verification['dotime_full']['episodes']==12150 and verification['source_pins_verified']
 inverse=json.loads((ROOT/'outputs/inverse_score/results.json').read_text(encoding='utf-8'));assert len(inverse['rows'])==6
 literature=json.loads((ROOT/'outputs/literature/registry.json').read_text(encoding='utf-8'));assert len(literature['papers'])==5 and not literature['errors']
 for row in literature['papers']:assert sha(ROOT/row['path'])==row['sha256']
 download=json.loads((ROOT/'outputs/dotime_checkpoint_download.json').read_text(encoding='utf-8'));assert download['verified'] and sha(ROOT/'data/do_over_time_pfn_best.pt')==download['sha256']
 for key in ['ep_idm','acro','ac_mtm']:
  pin=json.loads((ROOT/f'sources/{key}/SOURCE_PIN.json').read_text(encoding='utf-8'))
  for row in pin['files']:assert sha(ROOT/'sources'/key/row['path'])==row['sha256']
 pin=json.loads((ROOT/'sources/dotime/RETAINED_MANIFEST.json').read_text(encoding='utf-8'))
 for row in pin['files']:assert sha(ROOT/'sources/dotime'/row['path'])==row['sha256']
 for name in ['literature_review.md','baseline_registry.md','experimental_results.md','negative_evidence.md','hypothesis_space.md','research_ideas.md','final_research_direction.md']:assert (ROOT/name).is_file()
 files=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='ARTIFACT_MANIFEST.json' and not p.name.endswith(('.partial','.download','.tmp'))]
 rows=[{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(files)]
 manifest={'root_directory':'cycle9','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':rows,'count':len(rows),'bytes':sum(r['bytes'] for r in rows),'objective_achieved':False,'decision':'Native inverse-score audit and full corrected DoTime-suite baseline executed; known controls/defects preclude a final novel direction','exclusions':['Bytecode and incomplete transfers; project runtime and original read-only upstream dependency lie outside this cycle, with exact native package copy retained inside']}
 (ROOT/'ARTIFACT_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
 for row in rows:assert sha(ROOT/row['path'])==row['sha256']
 print(json.dumps({'count':len(rows),'bytes':manifest['bytes'],'all_server_hashes_verified':True,'objective_achieved':False}))
if __name__=='__main__':main()
