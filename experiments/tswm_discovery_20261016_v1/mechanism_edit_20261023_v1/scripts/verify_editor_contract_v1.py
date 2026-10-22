"""Independent Torch CPU inspection of recorded context tensors, not model predictions."""
import pathlib,json,hashlib,datetime,csv
import numpy as np,torch
torch.set_num_threads(2)
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'editor_contract_recovery_v1'
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def dump(n,j):
 with (OUT/n).open('x',encoding='utf-8') as f:json.dump(j,f,indent=2,allow_nan=False)
j=read(OUT/'RESULTS_V1.json');assert j['status']=='NATIVE_RECORDER_CLOSURES_EXECUTED_WITH_CONTEXT_ROUTING_SPIES' and not j['learned_model_executed']
assert sha(j['payload']['path'])==j['payload']['sha256'];assert sha(ROOT/'SOURCE_EDITOR_CONTRACT_PROTOCOL_V1.md')==j['protocol_sha256']
assert read(ROOT/'editor_contract_execution_recovery_v1/exit.json')['exit_code']==0
q=read(ROOT/'SOURCE_ACQUISITION_V1.json');s=pathlib.Path(q['source_root']);manifest=read(ROOT/'SOURCE_AST_AUDIT_V1.json')
for x in manifest['files']:assert sha(s/x['path'])==x['sha256']
counts=0;summary=[];max_error=0.
with np.load(j['payload']['path'],allow_pickle=False) as z:
 for r in j['records']:
  key=pathlib.Path(r['file']).stem+'_'+r['fixture']+'_B'+str(r['batch']);contexts=torch.from_numpy(z[key+'_contexts']);base=torch.from_numpy(z[key+'_base_context']);delta=contexts-base[:,:contexts.shape[2]].unsqueeze(0);assert torch.isfinite(delta).all()
  for idx,d in enumerate(delta):
   mask=d!=0;row=r['edits'][idx];expected=dict(changed_batches=torch.where(mask.any(dim=(1,2)))[0].tolist(),changed_times=torch.where(mask.any(dim=(0,2)))[0].tolist(),changed_dimensions=torch.where(mask.any(dim=(0,1)))[0].tolist(),nonzero_elements=int(mask.sum()));assert all(row[k]==v for k,v in expected.items());counts+=4
   for k,val in [('max_abs',d.abs().max().item()),('time_variation',(d-d[:,:1]).abs().max().item())]:
    error=abs(row[k]-val);assert error<=1e-6;max_error=max(max_error,error);counts+=1
  nonzero=[x for x in r['edits'] if x['nonzero_elements']>0];name=r['file'];kind=r['fixture']
  if kind=='native_broadcast' and name in ['record_counterfactual_plausibility.py','record_counterfactual_plausibility_obs.py','record_counterfactual_plausibility_action.py']:assert not nonzero
  if kind=='axis_only_non_native' and name=='record_counterfactual_plausibility.py':assert nonzero and all(x['changed_times']==[6] and x['changed_dimensions']==list(range(8)) for x in nonzero)
  if kind=='native_broadcast' and name=='record_counterfactual_plausibility_obs_dataset.py':assert len(nonzero)==8 and all(x['changed_dimensions']==[i] and x['time_variation']>0 for i,x in enumerate(nonzero))
  if name=='record_counterfactual_dreams.py':assert len(nonzero)==32 and all(x['changed_batches']==[0] and x['time_variation']==0 and len(x['changed_dimensions'])==1 for x in nonzero)
  summary.append(dict(file=name,fixture=kind,batch=r['batch'],input_length=r['input_length'],context_calls=len(r['edits']),nonzero_context_calls=len(nonzero),maximum_edit=max(x['max_abs'] for x in r['edits']),maximum_edit_time_variation=max(x['time_variation'] for x in r['edits']),root_indices=sorted({int(x) for a in r['roots'] for x in a}),learned_model_executed=False))
 for B in [1,2]:
  x=torch.from_numpy(z['control_B'+str(B)]);assert torch.all(x[:,:,6]==.5) and torch.count_nonzero(x[:,:,:6])==0 and torch.count_nonzero(x[:,:,7:])==0;counts+=3
assert len(j['records'])==18 and len(j['source_contracts'])==5
with (OUT/'CASE_SUMMARY_V1.csv').open('x',encoding='utf-8',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
dump('VERIFICATION_V1.json',dict(status='RECORDED_CONTEXT_EDIT_AXES_AND_TIME_CONTRACTS_INDEPENDENTLY_VERIFIED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),implementation='Torch CPU; worker summaries used NumPy and original edits executed in JAX',checks=counts,max_scalar_error=max_error,cases=summary,source_files_rehashed=len(manifest['files']),results_sha256=sha(OUT/'RESULTS_V1.json'),payload_sha256=j['payload']['sha256'],learned_model_executed=False,optimizer_updates=0,accepted_final_directions=0))
print(json.dumps(dict(status='RECORDED_EDITOR_CONTRACT_VERIFIED',checks=counts,max_scalar_error=max_error,cases=summary)),flush=True)
