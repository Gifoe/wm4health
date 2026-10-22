"""Torch float64 independent replay of common-noise device controls and source boundaries."""
import pathlib,json,hashlib,datetime
import numpy as np,torch
torch.set_num_threads(2);ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'native_verification_v1'
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
r=read(ROOT/'inference_parity_v1/RESULTS_V1.json');cases={x['tag']:x for x in r['cases']};arrays={};metric_errors=[]
for tag,c in cases.items():
 assert sha(c['payload_path'])==c['payload_sha256'] and sha(c['bank_path'])==c['bank_sha256'] and c['gaussian_arrays']==251
 with np.load(c['payload_path'],allow_pickle=False) as z:arrays[tag]={k:z[k].copy() for k in z.files}
 a=arrays[tag];x=torch.from_numpy(a['x']).double();y=torch.from_numpy(a['x_pred']).double();assert x.shape==(5,125,50) and torch.isfinite(y).all()
 for key,lo,hi in [('recon_mse',0,75),('pred_mse',75,125)]:
  score=float(((x[:,lo:hi]-y[:,lo:hi])**2).sum(-1).mean());error=abs(score-c['native_metrics'][key]);assert error<=1e-5+2e-6*abs(score);metric_errors.append(error)
comparison_checks=[]
for row in r['comparisons']:
 a=arrays[row['left']];b=arrays[row['right']];assert np.array_equal(a['x'],b['x']);x=torch.from_numpy(a['x_pred']).double();y=torch.from_numpy(b['x_pred']).double();ratio=float((x-y).square().mean().sqrt()/x.square().mean().sqrt());maximum=float((x-y).abs().max());assert abs(ratio-row['relative_prediction_rms'])<1e-12 and maximum==row['prediction_max'];comparison_checks.append(dict(left=row['left'],right=row['right'],relative_rms=ratio,maximum=maximum))
def bank(tag):
 with np.load(cases[tag]['bank_path'],allow_pickle=False) as z:return [z[k].copy() for k in sorted(z.files)]
a=bank('cpu24_native_s1000');b=bank('cpu28_native_s1000');c=bank('gpu28_native_s1000');assert len(a)==len(b)==len(c)==251 and all(np.array_equal(x,y) for x,y in zip(a,b));draw_components=sum(x.size for x in b);draw_rms=float(np.sqrt(sum(np.square(x.astype(np.float64)-y.astype(np.float64)).sum() for x,y in zip(b,c))/draw_components));assert draw_rms>0
assert cases['gpu28_cpu_bank_s1000']['bank_sha256']==cases['cpu28_native_s1000']['bank_sha256'] and cases['cpu28_gpu_bank_s1000']['bank_sha256']==cases['gpu28_native_s1000']['bank_sha256']
# Independently check the stored original replay and future-observation qualification outputs.
native=read(ROOT/'native_recipe_v1/RESULTS_V1.json');storage=read(ROOT.parent/'STORAGE_LOCATION_V1.json');art=pathlib.Path(storage['artifact_root'])/ROOT.name/'native_recipe_v1';replay_checks={}
for name,other,keys in [('meta_ilad_id_s42','replay_meta_ilad_id_s42',['x','x_pred','x_recon']),('meta_ilad_ood_s42','future_boundary_meta_ilad_ood_s42',['x_pred','x_recon'])]:
 base=next(x for x in native['cells'] if x['name']==name)
 with np.load(base['payload_path'],allow_pickle=False) as x,np.load(art/(other+'.npz'),allow_pickle=False) as y:
  for key in keys:
   delta=float(np.max(abs(x[key].astype(np.float64)-y[key].astype(np.float64))));assert delta==0;replay_checks[name+'/'+key]=delta
  if 'future_boundary' in other:assert np.array_equal(x['x'][:,:75],y['x'][:,:75]) and not np.array_equal(x['x'][:,75:],y['x'][:,75:])
out=dict(status='IDF_DEVICE_NOISE_ATTRIBUTION_AND_INPUT_BOUNDARIES_INDEPENDENTLY_VERIFIED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cases=7,metric_checks=len(metric_errors),max_metric_error=max(metric_errors),comparisons=comparison_checks,gaussian_arrays_per_bank=251,gaussian_scalar_components_per_bank=draw_components,cpu24_vs_cpu28_draw_banks_exact=True,cpu_vs_gpu_native_draw_difference_rms=draw_rms,common_draw_bank_sha_links_verified=True,replay_and_boundary_checks=replay_checks,random_draw_explanation_supported_locally=r['random_draw_explanation_supported_locally'],all_gpu_reference_metrics_within_tolerance=r['all_gpu_reference_metrics_within_tolerance'],external_vendor_execution_uniquely_recovered=False,physical_mechanism_gap_established=False,accepted_final_directions=0)
with (OUT/'ATTRIBUTION_VERIFICATION_V1.json').open('x',encoding='utf-8') as f:json.dump(out,f,indent=2,allow_nan=False)
print(json.dumps(out),flush=True)
