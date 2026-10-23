"""Independent observed-context/query scoring, selection checks and fresh native replay."""
import pathlib,sys,json,hashlib,datetime,traceback
import torch,numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'joint_context_refinement_verification_v1';OUT.mkdir(exist_ok=False)
ART=pathlib.Path(r'G:\ICLR Omni-eeg cache\tswm_discovery_20261016_v1\artifacts\meta_ilad_native_20261023_v1\joint_context_refinement_v1/arrays.npz')
SRC=pathlib.Path(r'G:\ICLR Omni-eeg cache\tswm_discovery_20261016_v1\caches\meta_ilad_native_20261023_v1\source')
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
def sha_file(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(n,x):
 with (OUT/n).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def risk(p,t):
 diff=np.asarray(p,dtype=np.float64)-t;return np.einsum('nlhw,nlhw->n',diff,diff)/diff.shape[1]
try:
 torch.set_num_threads(1);r=read(ROOT/'joint_context_refinement_v1/RESULTS_V1.json');plan=read(ROOT/'joint_context_refinement_v1/PLAN_V1.json');receipt=read(ROOT/'joint_context_refinement_execution_v1/exit.json');assert receipt['exit_code']==0 and receipt['log_sha256']==sha_file(ROOT/'joint_context_refinement_execution_v1/stdout.log')
 assert sha_file(ART)==r['payload_sha256'];a=np.load(ART);oldplan=read(ROOT/'context_transplant_audit_v1/PLAN_V1.json');oldr=read(ROOT/'context_transplant_audit_v1/RESULTS_V1.json');oldpath=ART.parent.parent/'context_transplant_audit_v1/arrays.npz';assert sha_file(oldpath)==plan['old_payload_sha256']==oldr['payload_sha256'];old=np.load(oldpath)
 assert plan['rows']==oldplan['rows'];envs=sorted(map(int,oldplan['contexts']));assert a['environment_ids'].tolist()==plan['environment_ids']==envs;contexts=np.array([oldplan['contexts'][str(env)] for env in envs]);assert np.array_equal(contexts,a['context_indices']) and contexts.tolist()==plan['context_indices']
 si=old['source_indices'];ti=old['target_indices'];assert not set(np.r_[si,ti].tolist()).intersection(contexts.reshape(-1).tolist())
 data=next(x for x in read(ROOT/'pendulum_data_generation_v1/RESULTS_V1.json')['sets'] if x['split']=='test');assert sha_file(data['path'])==plan['data_sha256'];raw=torch.load(data['path'],weights_only=True,mmap=True,map_location='cpu');truth=raw['yt'].numpy();e=len(envs);context_truth=truth[contexts,:40].reshape(e*11,40,32,32).astype(np.float64)
 selection_errors=[];context_means={}
 for name,array,key in [('initial','initial_context_prediction',None),('roots_only','roots_only_context_prediction','roots_only'),('joint','joint_context_prediction','joint')]:
  losses=risk(a[array],context_truth).reshape(e,11).mean(1);context_means[name]=float(losses.mean());assert abs(context_means[name]-r['context_mse_'+name])<1e-5
  if key:
   trace=a[key+'_trace'];best=a[key+'_best_step'];stored=a[key+'_best_loss'];assert trace.shape==(101,e) and np.array_equal(best,np.argmin(trace,axis=0)) and np.array_equal(stored,trace[best,np.arange(e)]);selection_errors.append(float(np.max(np.abs(losses-stored))))
 assert max(selection_errors)<1e-5
 source=truth[si,:40].astype(np.float64);target=truth[ti,:40].astype(np.float64);effect=target[:,3:]-source[:,3:];trace=a['query_prefix_trace'];best=a['query_prefix_best_step'];assert trace.shape==(61,160) and np.array_equal(best,np.argmin(trace,axis=0)) and np.array_equal(a['query_prefix_best_loss'],trace[best,np.arange(160)])
 prefix_error=float(np.max(np.abs(risk(a['selected_query_prefix_prediction'],source[:,:3])-a['query_prefix_best_loss'])));assert prefix_error<1e-5
 scores={}
 for name,cf,fact in [('old_naive',old['naive_transplant_prediction'],old['factual_prediction']),('old_fit',old['shooting_transplant_prediction'],old['shooting_factual_prediction']),('refined_naive',a['refined_naive_prediction'],a['refined_factual_prediction']),('refined_fit',a['refined_fit_prediction'],a['refined_fit_factual_prediction'])]:
  scores[name+'_cf']=risk(cf[:,3:],target[:,3:]);scores[name+'_factual']=risk(fact[:,3:],source[:,3:]);scores[name+'_response']=risk(cf[:,3:].astype(np.float64)-fact[:,3:].astype(np.float64),effect)
 scores['old_target_prefix']=risk(old['target_prefix_diagnostic_prediction'][:,3:],target[:,3:]);scores['refined_target_prefix']=risk(a['refined_target_prefix_prediction'][:,3:],target[:,3:]);scores['zero_effect']=np.einsum('nlhw,nlhw->n',effect,effect)/37
 errors=[abs(float(v.mean())-r['aggregate'][k]) for k,v in scores.items()];pairs=[]
 for pair in range(20):
  means={k:float(v[pair*8:(pair+1)*8].mean()) for k,v in scores.items()};record=r['pairs'][pair]
  for k,v in means.items():errors.append(abs(v-record['metrics'][k]))
  ratio=means['refined_fit_cf']/means['refined_target_prefix'];response=means['refined_fit_response']/means['zero_effect'];assert bool(ratio<=1.25 and response<=.25)==record['within_descriptive_thresholds'];errors.extend([abs(ratio-record['refined_fit_ratio']),abs(response-record['refined_fit_response_ratio'])]);pairs.append(dict(pair=pair,source_environment=record['source_environment'],target_environment=record['target_environment'],metrics=means,refined_fit_ratio=ratio,refined_fit_response_ratio=response,within_descriptive_thresholds=bool(ratio<=1.25 and response<=.25)))
 assert max(errors)<1e-10,max(errors)
 # Fresh original-method replay checks actual saved predictions from the fitted latent variables.
 torch.backends.cudnn.deterministic=True;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;sys.path.insert(0,str(SRC/'pendulum/meta_ilad'));from model import iLaD
 ckpt=SRC/'pendulum/checkpoints/meta_ilad_s1000.pt';assert sha_file(ckpt)==plan['checkpoint_sha256'];model=iLaD('cuda').cuda();model.load_state_dict(torch.load(ckpt,weights_only=True,map_location='cuda'));model.eval();grid=torch.linspace(0.,3.9,40,device='cuda')
 def rollout(z,c):
  with torch.no_grad():return model.decode(model.dynamics(torch.from_numpy(z.copy()).cuda(),torch.from_numpy(c.copy()).cuda(),compute_logp=False,t=grid)).cpu().numpy()
 replay=[]
 for name,z,c,key in [('initial',a['context_initial_states'][0],np.repeat(a['initial_codes'][:1],11,0),'initial_context_prediction'),('roots_only',a['roots_only_states'][0],np.repeat(a['initial_codes'][:1],11,0),'roots_only_context_prediction'),('joint',a['joint_states'][0],np.repeat(a['refined_codes'][:1],11,0),'joint_context_prediction')]:
  err=float(np.max(np.abs(rollout(z,c)-a[key][:11])));assert err<1e-4,(name,err);replay.append(dict(path=name,max_pixel_error=err))
 target_env=plan['rows'][0]['target_environment'];code=a['refined_codes'][envs.index(target_env)];codes=np.repeat(code[None],8,axis=0)
 for name,z,key in [('query_naive',old['source_z0'][:8],'refined_naive_prediction'),('query_fit',a['refined_query_states'][:8],'refined_fit_prediction')]:
  err=float(np.max(np.abs(rollout(z,codes)-a[key][:8])));assert err<1e-4,(name,err);replay.append(dict(path=name,max_pixel_error=err))
 assert r['model_state_before']==r['model_state_after'];assert r['context_optimizer_updates']==200 and r['query_state_optimizer_updates']==60 and r['new_network_weight_training_fits']==0
 result=dict(status='JOINT_CONTEXT_REFINEMENT_METRICS_SELECTION_AND_NATIVE_REPLAY_VERIFIED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),query_metric_checks=len(errors),max_query_metric_error=max(errors),context_risks=context_means,max_context_selection_error=max(selection_errors),prefix_selection_error=prefix_error,context_sets_and_exclusions_exact=True,all_selection_uses_observed_context_or_query_prefix=True,fresh_original_forward_replays=replay,aggregate={k:float(v.mean()) for k,v in scores.items()},pairs=pairs,context_optimizer_updates=200,query_state_optimizer_updates=60,new_network_weight_training_fits=0,novel_direction_accepted=False,local_payload_sync=False)
 dump('VERIFICATION_V1.json',result);print(json.dumps(dict(status=result['status'],metric_checks=len(errors),max_query_metric_error=max(errors),residual_pairs=[x for x in pairs if x['pair'] in [5,6,8]],native_replays=replay)),flush=True)
except Exception as exc:
 dump('FAILURE_V1.json',dict(exception_type=type(exc).__name__,exception=str(exc),traceback=traceback.format_exc()));raise
