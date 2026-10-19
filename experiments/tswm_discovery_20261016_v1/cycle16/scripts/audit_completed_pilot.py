"""Verify completed pilot provenance, all saved checkpoints/splits/metrics and exact replay."""
import runtime_profile
import hashlib,json,math,pathlib
import numpy as np
import torch
from data_protocol import metrics
ROOT=pathlib.Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 train=ROOT/'outputs/train/NP_MTS3_seed2021';fit=json.loads((train/'results.json').read_text())
 status=json.loads((ROOT/'outputs/jobs/native_mts3_NP_seed2021_attempt1/status.json').read_text())
 cfg=json.loads((train/'config.json').read_text());reg=json.loads((ROOT/'outputs/data/registry.json').read_text())
 assert status['status']=='completed' and status['exit_code']==0 and fit['training_complete'] and fit['last_step']==1500
 assert fit['no_test_evaluation'] and not reg['test_labels_exported'] and not reg['test_labels_evaluated']
 assert sha(ROOT/'FULL_MODEL_PLAN.md')==cfg['plan_sha256'] and sha(ROOT/'outputs/data/registry.json')==cfg['data_registry_sha256']
 assert sha(ROOT/'scripts/train_full_pilot.py')==cfg['script_sha256']==status['script_sha256']
 for key,name in [('adapter_sha256','data_protocol.py'),('config_adapter_sha256','config_tools.py')]:assert sha(ROOT/'scripts'/name)==cfg[key]
 manifest=json.loads((ROOT.parent/'cycle15/outputs/source/manifest.json').read_text());source=ROOT.parent/'cycle15/upstream/MTS3'
 for row in manifest['files']:assert sha(source/row['path'])==row['sha256']
 steps=[json.loads(x) for x in (train/'training.jsonl').read_text().splitlines()]
 assert [r['step'] for r in steps]==list(range(1,1501))
 for row in steps:
  assert math.isfinite(row['NLL']) and math.isfinite(row['gradient_norm_before_clip']) and row['seconds']>0
  origins=np.asarray(row['origins']);assert len(origins)==cfg['batch_size'] and np.all(origins>=reg['history']-1) and np.all(origins+reg['horizon']<reg['train_end'])
 assert np.all(np.diff([r['elapsed_seconds'] for r in steps])>0)
 validations=[json.loads(x) for x in (train/'validation.jsonl').read_text().splitlines()]
 assert [r['step'] for r in validations]==list(range(0,1501,100)) and validations==fit['validation']
 metric_error=0.;checkpoints=[]
 for row in validations:
  step=row['step'];p=train/f'checkpoint_step{step:06d}.pt';cp=torch.load(p,map_location='cpu',weights_only=True)
  assert cp['step']==step and cp['seed']==2021 and cp['config_sha256']==sha(train/'config.json')
  assert all(torch.isfinite(v).all() for v in cp['model'].values())
  assert all(key in cp for key in ['optimizer','torch_rng_cpu','torch_rng_cuda','numpy_rng_json']);json.loads(cp['numpy_rng_json'])
  with np.load(train/f'validation_step{step:06d}.npz',allow_pickle=False) as a:
   want=metrics(a['mean'],a['variance'],a['target']);roots=a['origins']
   assert np.all(roots+1>=reg['train_end']) and np.all(roots+reg['horizon']<reg['val_end']) and np.all(np.diff(roots)>=reg['horizon'])
   for key,value in want.items():metric_error=max(metric_error,abs(value-row[key]));assert np.isclose(value,row[key],rtol=1e-12,atol=1e-12)
  checkpoints.append({'step':step,'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p)})
 best=min(validations,key=lambda r:r['NLL']);pointer=json.loads((train/'best_checkpoint.json').read_text())
 assert best['step']==fit['best_step']==pointer['step'] and sha(ROOT/pointer['path'])==pointer['sha256']
 diagnostic=ROOT/f"outputs/diagnostics/final_step{best['step']:06d}"
 verification=json.loads((diagnostic/'independent_verification.json').read_text());assert verification['passed']
 assert verification['results_sha256']==sha(diagnostic/'results.json') and verification['arrays_sha256']==sha(diagnostic/'arrays.npz')
 with np.load(diagnostic/'arrays.npz',allow_pickle=False) as a,np.load(train/f"validation_step{best['step']:06d}.npz",allow_pickle=False) as old:
  np.testing.assert_array_equal(a['origins'],old['origins']);np.testing.assert_array_equal(a['target'],old['target'])
  mean_error=float(np.max(np.abs(a['native_mean']-old['mean'])));var_error=float(np.max(np.abs(a['native_variance']-old['variance'])))
  np.testing.assert_allclose(a['native_mean'],old['mean'],rtol=1e-6,atol=1e-6);np.testing.assert_allclose(a['native_variance'],old['variance'],rtol=1e-6,atol=1e-6)
 result={'passed':True,'source_files_verified':len(manifest['files']),'training_updates_verified':len(steps),'checkpoints_verified':checkpoints,
         'validation_snapshots_verified':len(validations),'all_training_and_validation_roots_within_declared_splits':True,
         'maximum_saved_metric_error':metric_error,'replayed_mean_max_error':mean_error,'replayed_variance_max_error':var_error,
         'best_checkpoint_selection_verified':True,'test_evaluation_recorded':False,'script_sha256':sha(pathlib.Path(__file__)),
         'scope':'Completed-pilot integrity only; does not prove training optimality, generality or innovation','objective_achieved':False}
 out=ROOT/'outputs/analysis/completed_pilot_verification.json';assert not out.exists();out.parent.mkdir(parents=True,exist_ok=True)
 out.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='checkpoints_verified'}),flush=True)
if __name__=='__main__':main()
