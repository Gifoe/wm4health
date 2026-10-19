"""Seal one completed discriminator; do not mark research discovery complete."""
import datetime,hashlib,json,pathlib,zipfile
ROOT=pathlib.Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while chunk:=f.read(1048576):h.update(chunk)
 return h.hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
 assert not (ROOT/'ARTIFACT_MANIFEST.json').exists()
 result=read(ROOT/'outputs/native_chart/results.json')
 verified=read(ROOT/'outputs/native_chart_independent_verification.json')
 assert verified['verified'] and len(result['native_models'])==4 and len(verified['rows'])==12
 assert verified['results_sha256']==sha(ROOT/'outputs/native_chart/results.json')
 assert result['script_sha256']==sha(ROOT/'scripts/native_chart_audit.py')
 assert result['plan_sha256']==sha(ROOT/'NATIVE_CHART_AUDIT_PLAN.md')
 for row in verified['npz_files']:assert sha(ROOT/row['path'])==row['sha256']
 assert sha(ROOT/'outputs/native_worldts_executed_source.zip')==verified['source_zip_sha256']
 with zipfile.ZipFile(ROOT/'outputs/native_worldts_executed_source.zip') as bundle:
  for row in verified['native_source_files']:assert hashlib.sha256(bundle.read(row['path'])).hexdigest()==row['sha256']
 papers=read(ROOT/'outputs/literature/manifest.json')
 archived=[row for row in papers['sources'] if row['status']=='archived']
 assert len(archived)==5
 for row in archived:assert sha(ROOT/row['path'])==row['sha256']
 ba=read(ROOT/'outputs/bayesian_context_source/manifest.json')
 assert ba['revision']=='20cfb5230e7be2f980365a992dab138ddd027070' and ba['count']==36 and not ba['downloaded_source_executed']
 assert sha(ROOT/'outputs/bayesian_context_source/source.zip')==ba['source_zip_sha256']
 with zipfile.ZipFile(ROOT/'outputs/bayesian_context_source/source.zip') as bundle:
  for row in ba['files']:assert hashlib.sha256(bundle.read(row['path'])).hexdigest()==row['sha256']
 status=read(ROOT/'outputs/task_status.json')
 assert status['scientific_audit_complete'] and not status['objective_achieved'] and not status['novel_direction_accepted']
 for name in ['literature_review.md','baseline_registry.md','experimental_results.md','negative_evidence.md','hypothesis_space.md','research_ideas.md','final_research_direction.md']:assert (ROOT/name).is_file()
 external={};base=pathlib.Path('D:/baseline_reproduction_seed2021_v1')
 for model in result['native_models']:
  folder=base/'results/worldts'/f"{model['dataset']}_H{model['horizon']}_seed2021"
  inputs=[(folder/'stages/stage2_decoder_mix.pt',model['checkpoint_sha256']),
          (folder/'resolved_config.json',model['config_sha256']),
          (base/'data'/f"{model['dataset']}.csv",model['data_sha256'])]
  for path,expected in inputs:
   assert sha(path)==expected
   external[path.as_posix()]={'path':path.as_posix(),'bytes':path.stat().st_size,'sha256':expected,'scope':'Pre-existing read-only input; original location preserved'}
 files=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='ARTIFACT_MANIFEST.json' and not p.name.endswith(('.tmp','.partial','.download'))]
 rows=[{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(files)]
 payload={'root_directory':'cycle14','files':rows,'count':len(rows),'bytes':sum(row['bytes'] for row in rows),
          'external_readonly_inputs':list(external.values()),'scientific_discriminator_complete':True,'objective_achieved':False,'novel_direction_accepted':False,
          'decision':status['decision'],'archive_caveat':'Server GitHub fetch failed; official Bayesian baseline snapshot was fetched locally and SHA-verified after upload. Initial failed document request log retained.',
          'exclusions':['Bytecode, incomplete transfers, shared runtimes and unchanged original native checkpoints/data/configs'],
          'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 (ROOT/'ARTIFACT_MANIFEST.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
 print(json.dumps({'server_files_verified':len(rows),'bytes':payload['bytes'],'external_readonly_inputs_verified':len(external),'objective_achieved':False}))
if __name__=='__main__':main()
