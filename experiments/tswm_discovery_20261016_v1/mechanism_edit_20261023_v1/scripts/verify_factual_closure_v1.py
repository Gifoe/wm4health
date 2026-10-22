"""Independent closure accounting, clock-array replay and learned-projection check."""
import pathlib,json,hashlib,datetime
import numpy as np,torch
torch.set_num_threads(2);ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'factual_transfer_closure_20261023_v1';OUT.mkdir(exist_ok=False)
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
fit=read(ROOT/'factual_fit_recovery_v1/RESULTS_V1.json');post=read(ROOT/'postfit_queue_recovery_v1/RESULTS_V1.json');assert fit['optimizer_updates']==2000 and fit['total_actual_optimizer_calls']==2500 and post['clock_qualified'] and not fit['competence_gate'];assert read(ROOT/'factual_execution_recovery_v1/exit.json')['exit_code']==0
verification=[]
for step in [0,500,1000,2000]:
 p=ROOT/'factual_fit_recovery_v1'/f'VALIDATION_STEP{step:04d}_V1.json';r=read(p);v=read(ROOT/'factual_metric_verification_recovery_v1'/f'VERIFIED_STEP{step:04d}_V1.json');assert v['receipt_sha256']==sha(p) and sha(r['checkpoint_path'])==r['checkpoint_sha256'] and sha(r['payload_path'])==r['payload_sha256'];verification.append(v)
assert fit['selected']['step']==min(fit['validation_trace'],key=lambda x:x['mse'])['step']==2000
clock=post['clock_result'];assert sha(clock['payload_path'])==clock['payload_sha256'];checks=[]
with np.load(clock['payload_path'],allow_pickle=False) as a:
 for key in ['deter','logit','stoch']:
  for left,right in [('scanned_post','explicit_post'),('scanned_prior','explicit_prior')]:
   x=torch.from_numpy(a[left+'/'+key]).double();y=torch.from_numpy(a[right+'/'+key]).double();delta=float((x-y).abs().max());assert delta==0;checks.append(dict(comparison=left+'/'+key,maximum_error=delta))
  delta=float(np.max(abs(a['scanned_prior/'+key][:,:24].astype(np.float64)-a['changed_plan_prior/'+key][:,:24].astype(np.float64))));assert delta==0;checks.append(dict(comparison='nonanticipation/'+key,maximum_error=delta))
 for left,right,end in [('prediction','explicit_prediction',32),('prediction','changed_plan_prediction',24)]:
  delta=float(np.max(abs(a[left][:,:end].astype(np.float64)-a[right][:,:end].astype(np.float64))));assert delta==0;checks.append(dict(comparison=left+'/'+right,maximum_error=delta))
initial=fit['validation_trace'][0];chosen=fit['selected'];projection=read(ROOT/'crossmodal_checkpoint_comparisons_v1/STEP2000_V1.json')
with np.load(initial['checkpoint_path'],allow_pickle=False) as a,np.load(chosen['checkpoint_path'],allow_pickle=False) as b:
 forward={k:float((torch.from_numpy(a[k]).double()-torch.from_numpy(b[k]).double()).abs().max()) for k in a.files if '/recon_wm/' in k};reverse={k:float((torch.from_numpy(a[k]).double()-torch.from_numpy(b[k]).double()).abs().max()) for k in a.files if '/recon_ctxc/' in k};assert forward==projection['forward_projection_changes'] and reverse==projection['reverse_projection_changes']
source=read(ROOT/'SOURCE_ACQUISITION_V1.json');src=pathlib.Path(source['source_root']);manifest=read(ROOT/'SOURCE_AST_AUDIT_V1.json')
for row in manifest['files']:assert sha(src/row['path'])==row['sha256']
result=dict(status='NATIVE_FACTUAL_TRANSFER_CLOSURE_INDEPENDENTLY_VERIFIED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),intended_fit_updates=2000,original_incomplete_updates=500,actual_optimizer_calls=2500,completed_fit_trajectories=1,original_logged_prefix_matches=len(fit['replay_comparisons']),prefix_scalar_max=max(x['max_scalar_difference'] for x in fit['replay_comparisons']),scalar_metric_checks=sum(x['scalar_metric_checks'] for x in verification),max_metric_error=max(x['max_metric_error'] for x in verification),clock_array_comparisons=checks,clock_arrays_exact=True,projection_forward_changes=forward,projection_reverse_changes=reverse,source_files_rehashed=len(manifest['files']),factual_competence_gate=False,physical_parameter_edit_evaluated=False,full_research_goal_achieved=False,accepted_final_directions=0)
with (OUT/'VERIFICATION_V1.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
print(json.dumps(result),flush=True)
