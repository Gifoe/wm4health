"""Verify registered pair/context construction, prefix-only selection and all image metrics."""
import pathlib,json,hashlib,datetime,traceback
import torch,numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'context_transplant_verification_v1';OUT.mkdir(exist_ok=False)
ART=pathlib.Path(r'G:\ICLR Omni-eeg cache\tswm_discovery_20261016_v1\artifacts\meta_ilad_native_20261023_v1\context_transplant_audit_v1/arrays.npz')
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
def sha_file(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(n,x):
 with (OUT/n).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
try:
 torch.set_num_threads(1);r=read(ROOT/'context_transplant_audit_v1/RESULTS_V1.json');plan=read(ROOT/'context_transplant_audit_v1/PLAN_V1.json');exit=read(ROOT/'context_transplant_execution_v1/exit.json');assert exit['exit_code']==0 and exit['log_sha256']==sha_file(ROOT/'context_transplant_execution_v1/stdout.log')
 assert sha_file(ART)==r['payload_sha256'];a=np.load(ART);data=read(ROOT/'pendulum_data_generation_v1/RESULTS_V1.json');test=next(x for x in data['sets'] if x['split']=='test');assert sha_file(test['path'])==plan['data_sha256']==test['file_sha256']
 raw=torch.load(test['path'],weights_only=True,mmap=True,map_location='cpu');state=raw['xt'].numpy();labels=raw['label'].numpy().reshape(-1).astype(int);images=raw['yt'].numpy()
 # Recreate the legacy author context pools without using the model's data loader.
 legacy=np.random.RandomState(42);pools={}
 for env in range(75):
  ix=np.where(labels==env)[0];order=np.arange(len(ix));legacy.shuffle(order);pools[env]=ix[np.sort(order[:13])]
 rng=np.random.default_rng(202610232);sources=rng.permutation(75)[:20];expected=[]
 for pair,s in enumerate(sources):
  s=int(s);t=(s+37)%75;excluded=set((pools[s]//75).tolist()+(pools[t]//75).tolist());allowed=np.array([i for i in range(40) if i not in excluded]);roots=rng.permutation(allowed)[:8]
  for env in [s,t]:assert plan['contexts'][str(env)]==pools[env][:11].tolist()
  for initial in roots:expected.append(dict(pair=pair,source_environment=s,target_environment=t,initial_state_index=int(initial),source_index=int(initial)*75+s,target_index=int(initial)*75+t))
 assert expected==plan['rows'];si=np.array([x['source_index'] for x in expected]);ti=np.array([x['target_index'] for x in expected]);assert np.array_equal(a['source_indices'],si) and np.array_equal(a['target_indices'],ti)
 assert np.array_equal(state[si,0],state[ti,0]);assert r['model_state_before']==r['model_state_after']
 source=images[si,:40].astype(np.float64);target=images[ti,:40].astype(np.float64);effect=target[:,3:]-source[:,3:]
 def per_mse(pred,truth):
  err=np.asarray(pred,dtype=np.float64)-truth;return np.einsum('nlhw,nlhw->n',err,err)/err.shape[1]
 scores={}
 for k in ['factual_prediction','shooting_factual_prediction']:scores[k]=per_mse(a[k][:,3:],source[:,3:])
 for k in ['naive_transplant_prediction','target_prefix_diagnostic_prediction','shooting_transplant_prediction']:scores[k]=per_mse(a[k][:,3:],target[:,3:])
 scores['source_persistence']=per_mse(np.broadcast_to(source[:,2:3],source[:,3:].shape),source[:,3:]);scores['target_persistence']=per_mse(np.broadcast_to(target[:,2:3],target[:,3:].shape),target[:,3:])
 scores['zero_effect_error']=np.einsum('nlhw,nlhw->n',effect,effect)/37
 for name,cf,factual in [('naive','naive_transplant_prediction','factual_prediction'),('shooting','shooting_transplant_prediction','shooting_factual_prediction')]:scores[name+'_response_error']=per_mse(a[cf][:,3:].astype(np.float64)-a[factual][:,3:].astype(np.float64),effect)
 errors=[]
 for key,values in scores.items():
  errors.append(abs(float(values.mean())-r['aggregate'][key]))
  for i,value in enumerate(values):errors.append(abs(float(value)-r['rows'][i][key]))
 assert max(errors)<1e-10,max(errors)
 trace=a['prefix_loss_trace'];steps=a['selected_prefix_step'];loss=a['selected_prefix_loss'];assert trace.shape==(61,160) and np.array_equal(steps,np.argmin(trace,axis=0)) and np.array_equal(loss,trace[steps,np.arange(160)])
 prefix=per_mse(a['selected_prefix_prediction'],source[:,:3]);prefix_error=float(np.max(np.abs(prefix-loss)));assert prefix_error<1e-5,prefix_error
 mean={k:float(v.mean()) for k,v in scores.items()};nr=mean['naive_transplant_prediction']/mean['target_prefix_diagnostic_prediction'];sr=mean['shooting_transplant_prediction']/mean['target_prefix_diagnostic_prediction'];ne=mean['naive_response_error']/mean['zero_effect_error'];se=mean['shooting_response_error']/mean['zero_effect_error'];excess=mean['naive_transplant_prediction']-mean['target_prefix_diagnostic_prediction'];reduction=1-(mean['shooting_transplant_prediction']-mean['target_prefix_diagnostic_prediction'])/excess if excess>0 else None
 gate=mean['factual_prediction']<.9*mean['source_persistence'] and mean['target_prefix_diagnostic_prediction']<.9*mean['target_persistence']
 decision='AUDIT_POINT_VARIANTS_NOT_FACTUALLY_QUALIFIED' if not gate else ('REJECT_CANDIDATE_NATIVE_CONTEXT_TRANSPLANT_SUFFICES' if nr<=1.25 and ne<=.25 else ('ORDINARY_STATE_INFERENCE_EXPLAINS_TRANSPLANT_DEFICIT' if (sr<=1.25 and se<=.25) or (reduction is not None and reduction>=.75) else 'CANDIDATE_REQUIRES_ATTRIBUTION_AND_PRIOR_ART_NOT_ACCEPTED_NOVELTY'))
 assert decision==r['decision'];assert r['state_estimation_optimizer_updates']==60 and r['weight_training_fits']==0
 result=dict(status='REGISTERED_CONTEXT_TRANSPLANT_AUDIT_INDEPENDENTLY_VERIFIED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),decision=decision,metric_checks=len(errors),max_metric_error=max(errors),physical_initial_state_pairs_exact=160,context_pools_exact=True,query_and_twin_excluded_from_context=True,prefix_selection_argmins_exact=True,prefix_recomputed_max_error=prefix_error,aggregate=mean,point_variant_factual_gate=gate,naive_to_target_prefix_mse_ratio=nr,shooting_to_target_prefix_mse_ratio=sr,naive_normalized_response_error=ne,shooting_normalized_response_error=se,ordinary_state_fit_excess_mse_reduction=reduction,novel_direction_accepted=False,local_payload_sync=False)
 dump('VERIFICATION_V1.json',result);print(json.dumps(result),flush=True)
except Exception as exc:
 dump('FAILURE_V1.json',dict(exception_type=type(exc).__name__,exception=str(exc),traceback=traceback.format_exc()));raise
