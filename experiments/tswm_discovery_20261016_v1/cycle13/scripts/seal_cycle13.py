"""Seal verified evidence, not the still-unachieved discovery objective."""
import datetime,hashlib,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(1048576):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))

def main():
 assert not (ROOT/'ARTIFACT_MANIFEST.json').exists()
 v=read(ROOT/'outputs/evidence_verification.json')
 assert v['all_checks_passed'] and v['fits_replayed']==24 and v['query_groups_independently_verified']==40
 assert v['maximum_saved_vs_original_GPU_replay_error']==0
 assert len(v['precision_controls'])==24 and v['maximum_no_cudnn_TF32_vs_CPU_control_error']<2e-5
 for folder in ['acquisition_controls','residual_controls']:
  r=read(ROOT/'outputs'/folder/'results.json');assert len(r['rows'])==12 and r['scientific_jobs_complete']
  assert len(list((ROOT/'outputs'/folder).glob('*.pt')))==24
 for name in ['literature_review.md','baseline_registry.md','experimental_results.md','negative_evidence.md','hypothesis_space.md','research_ideas.md','final_research_direction.md']:
  assert (ROOT/name).is_file()
 papers=read(ROOT/'outputs/literature/registry.json');assert len(papers['papers'])==6 and not papers['errors']
 for r in papers['papers']:assert sha(ROOT/r['path'])==r['sha256']
 status=read(ROOT/'outputs/task_status.json');assert status['verified_fits']==24 and not status['objective_achieved']
 external=[]
 for r in read(ROOT/'outputs/data_registry.json')['datasets']:
  assert sha(ROOT/r['path'])==r['sha256']
  if r['path'].startswith('../'):external.append({'path':r['path'],'bytes':r['bytes'],'sha256':r['sha256'],'scope':'read-only source already retained in sealed cycle5'})
 qa={'figure':'outputs/figures/ordinary_controls.png','visually_inspected_locally':True,'axes_labels_legends_readable':True,'error_bars':'paired training-seed SD,not CI','objective_achieved':False}
 (ROOT/'outputs/figure_QA.json').write_text(json.dumps(qa,indent=2),encoding='utf-8')
 files=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='ARTIFACT_MANIFEST.json' and not p.name.endswith(('.partial','.download','.tmp'))]
 rows=[{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(files)]
 result={'root_directory':'cycle13','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':rows,'count':len(rows),'bytes':sum(r['bytes'] for r in rows),
  'external_readonly_inputs':external,'objective_achieved':False,'decision':'No new mechanism motivated by ordinary acquisition-policy controls;retain evidence,stop expanding this benchmark',
  'exclusions':['bytecode,incomplete transfers and project-wide runtimes;explicit read-only ETTh1dependency retained in prior sealed cycle5']}
 (ROOT/'ARTIFACT_MANIFEST.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
 for r in rows:assert sha(ROOT/r['path'])==r['sha256']
 print(json.dumps({'count':len(rows),'bytes':result['bytes'],'external_readonly_inputs_verified':len(external),'all_server_hashes_verified':True,'objective_achieved':False}))

if __name__=='__main__':main()
