"""Independent Torch float64 replay of completed factual checkpoint receipts."""
import pathlib,json,hashlib,argparse,datetime,sys
import numpy as np,torch
torch.set_num_threads(2)
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'factual_metric_verification_v1';OUT.mkdir(exist_ok=True)
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--step',required=True,type=int,choices=[0,500,1000,2000]);args=p.parse_args();dest=OUT/f'VERIFIED_STEP{args.step:04d}_V1.json';assert not dest.exists(),'Already verified; preserve prior receipt'
receipt=ROOT/'factual_fit_v1'/f'VALIDATION_STEP{args.step:04d}_V1.json';row=read(receipt);assert row['step']==args.step
assert sha(row['payload_path'])==row['payload_sha256'] and sha(row['checkpoint_path'])==row['checkpoint_sha256']
matrix=read(ROOT/'factual_fit_v1/MATRIX_V1.json');assert matrix['protocol_sha256']==sha(ROOT/'FACTUAL_FIT_PROTOCOL_V1.md') and matrix['intervention_or_eval_accessed'] is False
val=matrix['data']['val'];assert sha(val['path'])==val['sha256']
with np.load(row['payload_path'],allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
with np.load(val['path'],allow_pickle=False) as z:
 assert np.array_equal(a['history'],z['observations'][:,48:48+65]) and np.array_equal(a['truth'],z['observations'][:,48+65:48+97]) and np.array_equal(a['family'],z['family'])
assert a['predictions'].shape==(64,4,32,16) and a['mean_prediction'].shape==(64,32,16) and a['context'].shape==(64,64,8)
assert all(np.isfinite(x).all() for x in a.values());assert np.array_equal(a['context'],np.broadcast_to(a['context'][:,:1],a['context'].shape))
draws=torch.from_numpy(a['predictions']).double();pred=torch.from_numpy(a['mean_prediction']).double();truth=torch.from_numpy(a['truth']).double();hist=torch.from_numpy(a['history']).double();recon=torch.from_numpy(a['reconstruction']).double();checks=0;errors=[]
mean_delta=float((draws.mean(1)-pred).abs().max());assert mean_delta<1e-6
def compare(actual,expected):
 global checks
 error=abs(float(actual)-float(expected));errors.append(error);checks+=1;assert error<=2e-12,(actual,expected,error)
compare(((pred-truth)**2).mean(),row['mse'])
for f in [0,1]:
 ids=torch.from_numpy(a['family']==f);r=next(x for x in row['families'] if x['family']==f);mse=((pred[ids]-truth[ids])**2).mean();pers=((hist[ids,-1:]-truth[ids])**2).mean();dr=((draws[ids]-truth[ids,None])**2).mean();rec=((recon[ids]-hist[ids])**2).mean()
 for actual,key in [(mse,'mse'),(pers,'persistence_mse'),(dr,'per_draw_mse'),(rec,'reconstruction_mse'),(mse/pers,'persistence_ratio'),(mse/r['lagged_reference_mse'],'lagged_ratio')]:compare(actual,r[key])
 assert bool(mse<.9*pers and mse<=1.25*r['lagged_reference_mse'])==r['competence_gate']
with np.load(row['checkpoint_path'],allow_pickle=False) as z:
 assert all(np.isfinite(z[k]).all() for k in z.files)
 count=sum(z[k].size for k in z.files if 'model_opt' not in k);assert count==17052480
 optimizer_registered=any('model_opt' in k for k in z.files);assert optimizer_registered==(args.step>0)
result=dict(status='FACTUAL_ARRAYS_AND_METRICS_INDEPENDENTLY_VERIFIED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),step=args.step,scalar_metric_checks=checks,max_metric_error=max(errors),float64_draw_mean_vs_saved_float32_mean_max=mean_delta,source_window_and_target_match=True,context_constant_over_time=True,model_array_elements=count,optimizer_state_registered=optimizer_registered,receipt_sha256=sha(receipt),checkpoint_sha256=row['checkpoint_sha256'],payload_sha256=row['payload_sha256'],physical_edit_evaluated=False,alignment_explicit_step_check=False,independent_confirmation=False)
with dest.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
print(json.dumps(result),flush=True)
