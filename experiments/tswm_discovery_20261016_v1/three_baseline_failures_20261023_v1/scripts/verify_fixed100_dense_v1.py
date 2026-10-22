"""Independent origin-array check of the already frozen fixed100 models."""
import pathlib,json,hashlib,datetime
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs'
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
assert read(OUT/'fixed100_dense_v1_exit.json')['exit_code']==0
assert read(OUT/'fixed100_dense_controller_v1/RESULTS_V1.json')['status']=='TWELVE_FIXED100_DENSE_AUDITS_CLOSED'
reg=read(OUT/'matched_cap_v1/RESULTS_V1.json');rows=[];errors=[]
for mode in ('latent','raw'):
 p=OUT/('fixed100_dense_'+mode+'_v1/RESULTS_V1.json');j=read(p)
 assert j['status']=='SIX_FROZEN_MODELS_DENSE_HOURLY_AUDIT_COMPLETE' and j['source_registry_sha256']==sha(OUT/'matched_cap_v1/RESULTS_V1.json')
 assert [(x['horizon'],x['seed']) for x in j['records']]==[(h,s) for h in (336,720) for s in (42,43,44)]
 for x in j['records']:
  m=next(z for z in reg['rows'] if z['mode']==mode and z['seed']==x['seed'] and z['horizon']==x['horizon'])
  assert m['checkpoint_sha256']==x['source_checkpoint_sha256'] and sha(x['arrays']['path'])==x['arrays']['sha256']
  with np.load(x['arrays']['path'],allow_pickle=False) as z:a=z['origin_mse'];b=z['origin_mae'];hour=z['forecast_hour'];ids=z['raw_target_origin']
  assert np.isfinite(a).all() and np.isfinite(b).all() and len(a)==x['dense_origins'] and np.all(np.diff(ids)==1)
  for actual,ref in [(float(a.mean()),x['dense_mse']),(float(b.mean()),x['dense_mae'])]:errors.append(abs(actual-ref))
  for phase in x['phase_scores']:
   mask=np.arange(len(a))%60==phase['phase'];assert int(mask.sum())==phase['n'];errors.append(abs(float(a[mask].mean())-phase['mse']))
  for hs in x['hour_scores']:
   mask=hour==hs['hour'];assert int(mask.sum())==hs['n'];errors.append(abs(float(a[mask].mean())-hs['mse']))
  assert x['native_prediction_max_abs_error']<=x['native_prediction_qualification_bound'] and x['raw_target_max_abs_error']<2e-9
  rows.append(dict(mode=mode,seed=x['seed'],horizon=x['horizon'],native_mse=m['reproduced_mse'],dense_mse=x['dense_mse'],dense_mae=x['dense_mae'],dense_origins=len(a),dense_better_than_native=x['dense_mse']<m['reproduced_mse']))
assert max(errors)<2e-11
summary=[]
for h in (336,720):
 q=dict(horizon=h)
 for mode in ('latent','raw'):q[mode+'_dense_mean']=float(np.mean([x['dense_mse'] for x in rows if x['mode']==mode and x['horizon']==h]))
 q['raw_relative_reduction']=1-q['raw_dense_mean']/q['latent_dense_mean'];summary.append(q)
dest=OUT/'fixed100_dense_verification_v1';dest.mkdir(exist_ok=False)
obj=dict(status='TWELVE_FIXED100_DENSE_MODELS_VERIFIED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),table=rows,summary=summary,max_scalar_error=max(errors),scalar_checks=len(errors),overlapping_windows_not_independent=True,new_training=False,new_method=False)
(dest/'verification.json').write_text(json.dumps(obj,indent=2),encoding='utf-8')
print(json.dumps(dict(status=obj['status'],summary=summary,dense_improved_models=sum(x['dense_better_than_native'] for x in rows),max_scalar_error=max(errors))),flush=True)
