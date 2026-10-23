"""Independent full-image scoring, context-set reconstruction and affine-head verification."""
import pathlib,json,hashlib,datetime,traceback
import torch,numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'context_resampling_verification_v1';OUT.mkdir(exist_ok=False)
ART=pathlib.Path(r'G:\ICLR Omni-eeg cache\tswm_discovery_20261016_v1\artifacts\meta_ilad_native_20261023_v1\context_resampling_audit_v1')
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
def sha_file(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(n,x):
 with (OUT/n).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
try:
 torch.set_num_threads(1);r=read(ROOT/'context_resampling_audit_v1/RESULTS_V1.json');plan=read(ROOT/'context_resampling_audit_v1/PLAN_V1.json');exit=read(ROOT/'context_resampling_execution_v1/exit.json');assert exit['exit_code']==0 and exit['log_sha256']==sha_file(ROOT/'context_resampling_execution_v1/stdout.log')
 for name,item in r['payloads'].items():assert sha_file(ART/name)==item['sha256']
 assert r['model_state_before']==r['model_state_after'];assert r['new_weight_training_fits']==r['new_state_estimation_updates']==0
 oldplan=read(ROOT/'context_transplant_audit_v1/PLAN_V1.json');oldr=read(ROOT/'context_transplant_audit_v1/RESULTS_V1.json');assert plan['rows']==oldplan['rows'] and plan['checkpoint_sha256']==oldplan['checkpoint_sha256']
 oldfile=ART.parent/'context_transplant_audit_v1/arrays.npz';assert sha_file(oldfile)==plan['old_payload_sha256']==oldr['payload_sha256'];old=np.load(oldfile);a=np.load(ART/'metadata.npz')
 test=next(x for x in read(ROOT/'pendulum_data_generation_v1/RESULTS_V1.json')['sets'] if x['split']=='test');assert sha_file(test['path'])==plan['data_sha256'];raw=torch.load(test['path'],weights_only=True,mmap=True,map_location='cpu')
 labels=raw['label'].numpy().reshape(-1).astype(int);rows=plan['rows'];si=np.array([x['source_index'] for x in rows]);ti=np.array([x['target_index'] for x in rows]);assert np.array_equal(a['source_indices'],si) and np.array_equal(a['target_indices'],ti)
 assert np.array_equal(raw['xt'][si,0].numpy(),raw['xt'][ti,0].numpy())
 allowed=[];legacy=[]
 for pair in range(20):
  rr=rows[pair*8:(pair+1)*8];roots={x['initial_state_index'] for x in rr};lib=[];orig=[]
  for env in [rr[0]['source_environment'],rr[0]['target_environment']]:
   ix=np.where(labels==env)[0];pool=[int(i) for i in ix if i//75 not in roots];assert len(pool)==32;lib.append(pool);orig.append(oldplan['contexts'][str(env)]);assert set(orig[-1]).issubset(pool)
  allowed.append(lib);legacy.append(orig)
 sets=[legacy,allowed];rng=np.random.default_rng(202610233)
 for draw in range(64):sets.append([[sorted(rng.choice(pool,11,replace=False).tolist()) for pool in pair] for pair in allowed])
 assert sets==plan['context_sets'] and plan['group_labels']==['legacy11','extra_information32']+[f'fixed11_draw{i:02d}' for i in range(64)]
 h=a['cached_embeddings'].astype(np.float64);code_errors=[]
 for group in range(66):
  for pair in range(20):
   for role in range(2):
    mean=h[sets[group][pair][role]].mean(0);code=mean@a['mu_head_weight'].astype(np.float64).T+a['mu_head_bias'];lv=mean@a['logvar_head_weight'].astype(np.float64).T+a['logvar_head_bias'];actual=a['source_codes' if role==0 else 'target_codes'][group,pair*8:(pair+1)*8]
    assert np.array_equal(actual,np.repeat(actual[:1],8,axis=0));code_errors.extend(np.abs(actual[0]-code).tolist());code_errors.extend(np.abs(a['native_context_logvars'][group,pair,role]-lv).tolist())
 assert max(code_errors)<=1e-5,max(code_errors)
 truth_s=raw['yt'][si,:40].numpy().astype(np.float64);truth_t=raw['yt'][ti,:40].numpy().astype(np.float64);effect=truth_t[:,3:]-truth_s[:,3:]
 fixed_fact=old['factual_prediction'].astype(np.float64);fixed_fit=old['shooting_factual_prediction'].astype(np.float64)
 stores={name:np.load(ART/(name+'.npy'),mmap_mode='r') for name in r['paths']};assert all(x.shape==(66,160,40,32,32) for x in stores.values())
 recomputed=np.empty_like(a['metrics']);metric_error=0.;qual=read(ROOT/'context_resampling_audit_v1/CACHE_AND_REPLAY_QUALIFICATION_V1.json');pixel_checks={}
 def risk(p,t):
  diff=np.asarray(p,dtype=np.float64)-t;return np.einsum('nlhw,nlhw->n',diff,diff)/diff.shape[1]
 for group in range(66):
  fact,cf,target,fit=[stores[name][group] for name in r['paths']];assert all(np.isfinite(x).all() for x in [fact,cf,target,fit])
  values=[risk(fact[:,3:],truth_s[:,3:]),risk(cf[:,3:],truth_t[:,3:]),risk(target[:,3:],truth_t[:,3:]),risk(fit[:,3:],truth_t[:,3:]),risk(cf[:,3:].astype(np.float64)-fixed_fact[:,3:],effect),risk(cf[:,3:].astype(np.float64)-fact[:,3:].astype(np.float64),effect),risk(fit[:,3:].astype(np.float64)-fixed_fit[:,3:],effect)]
  recomputed[group]=np.stack(values,axis=-1);metric_error=max(metric_error,float(np.max(np.abs(recomputed[group]-a['metrics'][group]))))
  if group==0:
   for name,key in zip(r['paths'],['factual_prediction','naive_transplant_prediction','target_prefix_diagnostic_prediction','shooting_transplant_prediction']):pixel_checks[name]=float(np.max(np.abs(stores[name][0]-old[key])));assert pixel_checks[name]==qual['legacy_pixel_max_errors'][name]
  if group%16==0:print(json.dumps(dict(stage='full_prediction_group_rescored',group=group,max_error=metric_error)),flush=True)
 assert metric_error<1e-9,metric_error
 zero=np.einsum('nlhw,nlhw->n',effect,effect)/37;assert np.max(np.abs(zero-a['zero_effect_error']))<1e-10
 summary_error=[];summaries=[]
 for pair in range(20):
  pm=recomputed[:,pair*8:(pair+1)*8].mean(1);den=float(zero[pair*8:(pair+1)*8].mean());vectors=[pm[:,1]/pm[:,2],pm[:,3]/pm[:,2],pm[:,4]/den,pm[:,5]/den,pm[:,6]/den];names=['naive_ratio','frozen_fit_ratio','naive_target_only_response','naive_both_contexts_response','frozen_fit_target_only_response'];row=r['summaries'][pair];assert row['pair']==pair
  for name,v in zip(names,vectors):
   expected=dict(legacy=float(v[0]),extra32=float(v[1]),fixed11_mean=float(v[2:].mean()),fixed11_median=float(np.median(v[2:])),fixed11_q10=float(np.percentile(v[2:],10)),fixed11_q90=float(np.percentile(v[2:],90)),legacy_empirical_percentile=float(np.count_nonzero(v[2:]<=v[0])/64))
   for k,value in expected.items():summary_error.append(abs(value-row[name][k]))
  ratio,fratio,response,paired,fresponse=vectors;naive_count=int(np.sum((ratio[2:]>1.25)|(response[2:]>.25)));fit_count=int(np.sum((fratio[2:]>1.25)|(fresponse[2:]>.25)));assert naive_count/64==row['fixed11_naive_exceedance_fraction'] and fit_count/64==row['fixed11_frozen_fit_exceedance_fraction'];assert bool(fratio[1]<=1.25 and fresponse[1]<=.25)==row['extra32_frozen_fit_within_descriptive_thresholds']
  summaries.append(dict(pair=pair,source_environment=row['source_environment'],target_environment=row['target_environment'],zero_effect_error=den,legacy_naive_ratio=float(ratio[0]),extra32_naive_ratio=float(ratio[1]),legacy_frozen_fit_ratio=float(fratio[0]),extra32_frozen_fit_ratio=float(fratio[1]),legacy_frozen_fit_response=float(fresponse[0]),extra32_frozen_fit_response=float(fresponse[1]),fixed11_naive_exceedances=naive_count,fixed11_frozen_fit_exceedances=fit_count,draws=64,extra32_frozen_fit_within_descriptive_thresholds=bool(fratio[1]<=1.25 and fresponse[1]<=.25)))
 assert max(summary_error)<1e-9,max(summary_error)
 result=dict(status='FULL_CONTEXT_RESAMPLING_DATA_AND_METRICS_INDEPENDENTLY_VERIFIED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),groups_verified=66,image_paths_verified=4,root_rows=160,raw_image_metric_checks=recomputed.size,max_metric_error=metric_error,affine_head_component_checks=len(code_errors),max_affine_head_error=max(code_errors),summary_component_checks=len(summary_error),max_summary_error=max(summary_error),context_sets_exact=True,query_and_twin_exclusions_exact=True,legacy_pixel_replay_checks=pixel_checks,summaries=summaries,new_weight_training_fits=0,new_state_estimation_updates=0,novel_direction_accepted=False,local_payload_sync=False)
 dump('VERIFICATION_V1.json',result);print(json.dumps(dict(status=result['status'],metric_checks=recomputed.size,max_metric_error=metric_error,affine_head_max_error=max(code_errors),residual_pairs=[x for x in summaries if x['pair'] in [5,6,8]])),flush=True)
except Exception as exc:
 dump('FAILURE_V1.json',dict(exception_type=type(exc).__name__,exception=str(exc),traceback=traceback.format_exc()));raise
