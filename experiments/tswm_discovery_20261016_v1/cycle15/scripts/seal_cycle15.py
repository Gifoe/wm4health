"""Seal completed native/control evidence, retaining all scope limits."""
import pathlib,json,hashlib,datetime,zipfile
ROOT=pathlib.Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(1048576):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
 assert not (ROOT/'ARTIFACT_MANIFEST.json').exists()
 v=read(ROOT/'outputs/independent_verification.json');r=read(ROOT/'outputs/operator/results.json');c=read(ROOT/'outputs/cost/results.json')
 assert v['all_arithmetic_checks_passed'] and v['native_rows']==630 and v['cost_rows_checked']==6
 assert v['expected_variance_inside_intervals_count']==v['total_variance_intervals']==168
 assert sha(ROOT/'outputs/operator/results.json')==v['operator_results_sha256']
 assert sha(ROOT/'outputs/cost/results.json')==v['cost_results_sha256']
 assert sha(ROOT/'PERSISTENT_CONTEXT_PLAN.md')==r['plan_sha256']
 assert sha(ROOT/'scripts/persistent_context_operator.py')==r['script_sha256']
 assert sha(ROOT/'ORDINARY_CONTROL_COST_PLAN.md')==c['plan_sha256']
 assert len(list((ROOT/'outputs/operator').glob('mc_case*.npz')))==6
 for name,count,revision in [('source',152,'ab971c0ec43df7664c4b727ca69d5b8dd0cab5da'),('hiprssm_source',16,'f4adef0ce5ed4b37c9847973df320f72965bd330')]:
  m=read(ROOT/'outputs'/name/'manifest.json');assert m['count']==count and m['revision']==revision and not m['excluded_files'] and not m['AGENTS_files']
  archive=ROOT/'outputs'/name/'source.zip';assert sha(archive)==m['source_zip_sha256']
  with zipfile.ZipFile(archive) as bundle:
   for row in m['files']:
    assert hashlib.sha256(bundle.read(row['path'])).hexdigest()==row['sha256']
    if name=='source':assert sha(ROOT/'upstream/MTS3'/row['path'])==row['sha256']
 papers=read(ROOT/'outputs/literature/manifest.json');assert len(papers['papers'])==4
 for row in papers['papers']:assert row['status']=='archived' and sha(ROOT/row['path'])==row['sha256']
 for name in ['literature_review.md','baseline_registry.md','experimental_results.md','negative_evidence.md','hypothesis_space.md','research_ideas.md','final_research_direction.md']:assert (ROOT/name).is_file()
 s=read(ROOT/'outputs/task_status.json');assert s['native_operator_discriminator_complete'] and not s['objective_achieved'] and not s['novel_direction_accepted']
 receipts={'source':'Observed successful tool command exits plus persisted numeric outputs; not a complete redirected stdout transcript',
           'commands':[{'script':'persistent_context_operator.py','exit_code':0,'rows':630,'MC_horizon_rows':42},
                       {'script':'ordinary_control_cost.py','exit_code':0,'cost_rows':6},{'script':'verify_operator.py','exit_code':0,'variance_intervals_verified':168}],
           'full_stdout_redirected':False,'raw_numeric_outputs_preserved':True,'training_fits':0}
 (ROOT/'outputs/execution_receipts.json').write_text(json.dumps(receipts,indent=2),encoding='utf-8')
 files=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='ARTIFACT_MANIFEST.json' and not p.name.endswith(('.download','.partial','.tmp')) and 'failed_attempt' not in p.name]
 rows=[{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(files)]
 result={'root_directory':'cycle15','files':rows,'count':len(rows),'bytes':sum(x['bytes'] for x in rows),
         'scientific_discriminator_complete':True,'new_training_fits':0,'native_trained_model_failure_established':False,
         'decision':s['decision'],'novel_direction_accepted':False,'objective_achieved':False,
         'exclusions':['Bytecode, incomplete optional data/source downloads, shared runtimes; no original checkpoints/data were changed'],
         'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 (ROOT/'ARTIFACT_MANIFEST.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
 for row in rows:assert sha(ROOT/row['path'])==row['sha256']
 print(json.dumps({'server_files_verified':len(rows),'bytes':result['bytes'],'objective_achieved':False}))
if __name__=='__main__':main()
