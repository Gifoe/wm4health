"""Seal completed private evidence; this never declares the full goal achieved."""
import datetime,hashlib,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  while data:=f.read(1048576):h.update(data)
 return h.hexdigest()
def main():
 assert not (ROOT/'ARTIFACT_MANIFEST.json').exists(),'Do not overwrite sealed evidence'
 status=json.loads((ROOT/'outputs/task_status.json').read_text());assert status['cycle8']['scientific_jobs_complete'] and not status['objective_achieved']
 exits=json.loads((ROOT/'outputs/process_exits.json').read_text());assert all(exits[k]==0 for k in ['data_audit','clock_rpm_controls','strong_transition_controls','timer_quantization'])
 literature=json.loads((ROOT/'outputs/literature/registry.json').read_text());assert len(literature['papers'])==10 and not literature['errors']
 for r in literature['papers']:assert sha(ROOT/r['path'])==r['sha256']
 pin=json.loads((ROOT/'sources/chamber/SOURCE_PIN.json').read_text());assert len(pin['files'])==42
 for r in pin['files']:assert sha(ROOT/'sources/chamber'/r['path'])==r['sha256']
 data=json.loads((ROOT/'outputs/data_download.json').read_text())
 for r in data['datasets']:assert sha(ROOT/'data'/(r['dataset']+'.zip'))==r['sha256']
 for name in ['literature_review.md','baseline_registry.md','experimental_results.md','negative_evidence.md','hypothesis_space.md','research_ideas.md','final_research_direction.md']:assert (ROOT/name).is_file()
 files=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='ARTIFACT_MANIFEST.json' and not p.name.endswith(('.partial','.download','.tmp'))]
 rows=[{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(files)];manifest={'root_directory':'cycle8','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':rows,'count':len(rows),'bytes':sum(r['bytes'] for r in rows),'exclusions':['Bytecode and incomplete transfers; runtime outside cycle8'],'objective_achieved':False,'decision':'Real-data qualification and ordinary controls complete; measured-label artifacts preclude accepted new-dynamics claim'}
 (ROOT/'ARTIFACT_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
 for r in rows:assert sha(ROOT/r['path'])==r['sha256']
 print(json.dumps({'count':len(rows),'bytes':manifest['bytes'],'all_server_hashes_verified':True,'objective_achieved':False}))
if __name__=='__main__':main()
