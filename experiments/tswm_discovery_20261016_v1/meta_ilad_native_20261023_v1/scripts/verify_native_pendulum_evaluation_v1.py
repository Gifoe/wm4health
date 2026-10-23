"""Independent float64 image/latent scoring of complete original-source executions."""
import pathlib,json,hashlib,datetime,itertools,traceback,sys
import torch,numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'pendulum_native_closure_v1';OUT.mkdir(exist_ok=False)
ART=pathlib.Path(r'G:\ICLR Omni-eeg cache\tswm_discovery_20261016_v1\artifacts\meta_ilad_native_20261023_v1\pendulum_native_recipe_v1')
SRC=pathlib.Path(r'G:\ICLR Omni-eeg cache\tswm_discovery_20261016_v1\caches\meta_ilad_native_20261023_v1\source')
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
def sha_file(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(n,x):
 with (OUT/n).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def mcc(x,y):
 x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64);x=x-x.mean(0);y=y-y.mean(0)
 corr=np.abs(x.T@y/np.sqrt((x*x).sum(0)[:,None]*(y*y).sum(0)[None,:]));d=x.shape[1]
 return float(max(sum(corr[i,j] for i,j in enumerate(p))/d for p in itertools.permutations(range(d))))
try:
 torch.set_num_threads(1);run=read(ROOT/'pendulum_native_recipe_v1/RESULTS_V1.json');assert run['complete_evaluator_invocations']==6
 receipt=read(ROOT/'pendulum_native_execution_v1/exit.json');assert receipt['exit_code']==0 and receipt['log_sha256']==sha_file(ROOT/'pendulum_native_execution_v1/stdout.log')
 data=read(ROOT/'pendulum_data_generation_v1/RESULTS_V1.json');rows=[];all_errors=[];reference=[];boundary_checks=[]
 for cell in run['cells']:
  name=cell['variant']+'_'+cell['split'];folder=ART/name;n=cell['sequences'];raw=next(x for x in data['sets'] if x['split']==cell['split']);assert sha_file(raw['path'])==raw['file_sha256']
  assert cell['loaded_state_digest']==cell['final_state_digest']
  for path,item in cell['payloads'].items():assert sha_file(folder/path)==item['sha256']
  target=torch.load(raw['path'],weights_only=True,mmap=True,map_location='cpu');pred=np.load(folder/'predictions.npy',mmap_mode='r');rec=np.load(folder/'reconstructions.npy',mmap_mode='r')
  assert pred.shape==rec.shape==(n,40,32,32);truth=target['yt'].numpy();native=np.load(folder/'native.npz')
  assert np.array_equal(native['z_true'],target['xt'][:,:40].numpy()) and np.array_equal(native['c_true'],target['params'].numpy())
  pred_seq=[];rec_seq=[];future=[];persistence=[]
  for start in range(0,n,64):
   x=truth[start:start+64,:40].astype(np.float64);p=pred[start:start+64].astype(np.float64);r=rec[start:start+64].astype(np.float64);assert np.isfinite(p).all() and np.isfinite(r).all()
   pred_seq.extend(np.square(x-p).sum((2,3)).mean(1).tolist());rec_seq.extend(np.square(x-r).sum((2,3)).mean(1).tolist())
   future.extend(np.square(x[:,3:]-p[:,3:]).sum((2,3)).mean(1).tolist());persistence.extend(np.square(x[:,3:]-x[:,2:3]).sum((2,3)).mean(1).tolist())
  def aggregate(v):return float(np.mean([np.mean(v[i:i+128]) for i in range(0,n,128)])) if cell['variant']=='paper' else float(np.mean(v))
  metrics=dict(recon_mse=aggregate(rec_seq),pred_mse=aggregate(pred_seq),recon_mcc_z=mcc(native['z_true'][...,0].reshape(-1,1),native['z_rec'][...,0].reshape(-1,1)),pred_mcc_z=mcc(native['z_true'][...,0].reshape(-1,1),native['z_pred'][...,0].reshape(-1,1)),recon_mcc_c=mcc(native['c_true'],native['c_rec']),pred_mcc_c=mcc(native['c_true'],native['c_pred']))
  errors={k:abs(v-cell['native_metrics'][k]) for k,v in metrics.items()};assert max(errors.values())<1e-5,(name,errors);all_errors.extend(errors.values())
  f=float(np.mean(future));baseline=float(np.mean(persistence));gate=f<.9*baseline and metrics['pred_mcc_z']>.99 and metrics['pred_mcc_c']>.95
  row=dict(cell=name,variant=cell['variant'],split=cell['split'],sequences=n,metrics_float64=metrics,native_metric_max_error=max(errors.values()),strict_future_mse=f,frame2_persistence_future_mse=baseline,competence_gate=gate,sequence_mean_pred_mse=float(np.mean(pred_seq)),unweighted_batch_mean_pred_mse=float(np.mean([np.mean(pred_seq[i:i+128]) for i in range(0,n,128)])))
  rows.append(row)
  if cell['variant']=='refined':
   ref=read(SRC/('pendulum/outputs/refined/saved/meta_ilad_s1000_'+cell['split']+'.json'))
   for k,v in metrics.items():
    rel=abs(v-ref[k])/max(abs(ref[k]),1e-12);reference.append(dict(cell=name,metric=k,value=v,reference=ref[k],relative_error=rel,within_frozen_one_percent=rel<=.01))
  if 'boundary_qualification' in cell:
   p=folder/'future_boundary_shadow.npz';assert sha_file(p)==cell['boundary_qualification']['payload_sha256'];z=np.load(p)
   assert np.array_equal(z['original_query'][:,:3],z['modified_query'][:,:3]) and not np.array_equal(z['original_query'][:,3:],z['modified_query'][:,3:])
   err=float(np.max(np.abs(z['original_prediction']-z['modified_prediction'])));assert err==0;boundary_checks.append(dict(cell=name,max_prediction_difference=err))
  print(json.dumps(dict(stage='cell_independently_verified',**row)),flush=True)
 comparisons=[]
 for split in ['test','ood','trainenv']:
  a=np.load(ART/('paper_'+split)/'predictions.npy',mmap_mode='r');b=np.load(ART/('refined_'+split)/'predictions.npy',mmap_mode='r');maximum=0.;squared=0.;norm=0.
  for i in range(0,len(a),64):
   x=a[i:i+64].astype(np.float64);y=b[i:i+64].astype(np.float64);maximum=max(maximum,float(np.max(np.abs(x-y))));squared+=float(np.square(x-y).sum());norm+=float(np.square(x).sum())
  comparisons.append(dict(split=split,paper_refined_max_pixel_difference=maximum,paper_refined_relative_rms=float(np.sqrt(squared/max(norm,1e-30)))))
 acq=read(ROOT/'pendulum_acquisition_recovery_v2/RESULTS_V1.json')
 for x in acq['files']:assert sha_file(SRC/x['path'])==x['sha256']
 result=dict(status='COMPLETE_NATIVE_PHYSICAL_METRICS_INDEPENDENTLY_VERIFIED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=rows,independent_native_metric_checks=len(all_errors),independent_max_native_metric_error=max(all_errors),refined_reference_checks=reference,all_refined_reference_metrics_within_one_percent=all(x['within_frozen_one_percent'] for x in reference),all_factual_competence_gates=all(x['competence_gate'] for x in rows),source_reference_bitwise_data_parity=all(x['source_bitwise_all_tensors'] for x in data['sets']),paper_refined_prediction_comparisons=comparisons,boundary_checks=boundary_checks,source_files_rehashed=len(acq['files']),complete_evaluator_invocations=6,additional_native_boundary_calls=2,new_fits=0,physical_mechanism_test_executed=False,novel_gap_established=False,local_payload_sync=False)
 dump('VERIFICATION_V1.json',result);print(json.dumps(dict(status=result['status'],all_refined_reference_metrics_within_one_percent=result['all_refined_reference_metrics_within_one_percent'],all_factual_competence_gates=result['all_factual_competence_gates'],max_metric_error=max(all_errors))),flush=True)
except Exception as exc:
 dump('FAILURE_V1.json',dict(exception_type=type(exc).__name__,exception=str(exc),traceback=traceback.format_exc()));raise
