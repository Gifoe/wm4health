"""Full independent raw-image scoring, leave-one-out selection and native replay."""
import pathlib,sys,json,hashlib,datetime,traceback
import numpy as np,torch
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'state_trust_region_verification_v1';OUT.mkdir(exist_ok=False)
ART=pathlib.Path(r'G:\ICLR Omni-eeg cache\tswm_discovery_20261016_v1\artifacts\meta_ilad_native_20261023_v1\state_trust_region_control_v1')
SRC=pathlib.Path(r'G:\ICLR Omni-eeg cache\tswm_discovery_20261016_v1\caches\meta_ilad_native_20261023_v1\source')
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
def sha_file(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(n,x):
 with (OUT/n).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def risk(p,t):return np.square(np.asarray(p,dtype=np.float64)-t).sum(axis=(2,3)).mean(axis=1)
try:
 torch.set_num_threads(1);r=read(ROOT/'state_trust_region_control_v1/RESULTS_V1.json');plan=read(ROOT/'state_trust_region_control_v1/PLAN_V1.json');selection=read(ROOT/'state_trust_region_control_v1/SELECTION_V1.json');receipt=read(ROOT/'state_guard_execution_v1/exit.json')
 assert receipt['exit_code']==0 and receipt['log_sha256']==sha_file(ROOT/'state_guard_execution_v1/stdout.log');assert plan['protocol_sha256']==sha_file(ROOT/'STATE_TRUST_REGION_CONTROL_PROTOCOL_V1.md');assert r['selection_sha256']==sha_file(ROOT/'state_trust_region_control_v1/SELECTION_V1.json')
 for x in r['payloads']:assert sha_file(ART/x['name'])==x['sha256']
 a=np.load(ART/'metadata_and_query_predictions.npz');pred=np.load(ART/'calibration_predictions.npy',mmap_mode='r');assert pred.shape==(8,5,825,20,32,32)
 oldplan=read(ROOT/'rolling_origin_audit_v1/PLAN_V1.json');assert sha_file(ROOT/'rolling_origin_audit_v1/PLAN_V1.json')==plan['old_plan_sha256'];contexts=np.asarray(oldplan['contexts']);q=np.asarray(oldplan['queries']);envs=np.asarray(oldplan['environment_ids']);assert np.array_equal(a['contexts'],contexts) and contexts.tolist()==plan['contexts'];assert np.array_equal(q,a['queries']) and q.tolist()==plan['queries'];assert np.array_equal(envs,a['environment_ids']) and envs.tolist()==plan['environment_ids'];assert not set(q).intersection(contexts.reshape(-1));assert plan['calibration_indices']==contexts.reshape(-1).tolist()==a['calibration_indices'].tolist()
 data=next(x for x in read(ROOT/'pendulum_data_generation_v1/RESULTS_V1.json')['sets'] if x['split']=='test');assert sha_file(data['path'])==plan['data_sha256'];raw=torch.load(data['path'],weights_only=True,mmap=True,map_location='cpu');truth=raw['yt'].numpy();calib_truth=truth[contexts.reshape(-1),:40].astype(np.float64);query_truth=truth[q,:40].astype(np.float64)
 candidates=[None,10.,1.,.1,.01,.001,.0001,0.];origins=[0,5,10,15,20];assert candidates==plan['candidates']==selection['candidates'] and origins==plan['origins']
 h=a['support_embeddings'].astype(np.float64);w=a['mu_head_weight'].astype(np.float64);b=a['mu_head_bias'].astype(np.float64);head_errors=[]
 for env in range(75):
  head_errors.extend(np.abs(h[env].mean(0)@w.T+b-a['full_environment_codes'][env]).tolist())
  for left in range(11):head_errors.extend(np.abs(np.delete(h[env],left,0).mean(0)@w.T+b-a['calibration_codes'][env*11+left]).tolist())
 assert max(head_errors)<1e-5
 prefix_errors=[];objective_errors=[];metric_errors=[];calib_risks=np.empty((8,5,825));checks=0
 for ci,coefficient in enumerate(candidates):
  for oi,origin in enumerate(origins):
   images=pred[ci,oi];assert np.isfinite(images).all();calib_risks[ci,oi]=risk(images[:,3:],calib_truth[:,origin+3:origin+20]);metric_errors.append(float(np.max(np.abs(calib_risks[ci,oi]-a['calibration_risks'][ci,oi]))));checks+=825
   mu=a['calibration_mu'][ci,oi];lv=a['calibration_logvar'][ci,oi];variance=a['calibration_variance'][ci,oi];state=a['calibration_state'][ci,oi];scale=a['calibration_scale'][ci,oi]
   assert np.array_equal(mu,a['calibration_mu'][0,oi]) and np.array_equal(lv,a['calibration_logvar'][0,oi]);assert np.allclose(variance,np.maximum(np.exp(lv.astype(np.float64)),1e-8),atol=1e-10,rtol=1e-6)
   trace=a['calibration_trace'][ci,oi];step=a['calibration_step'][ci,oi];best=a['calibration_best_loss'][ci,oi];assert trace.shape==(61,825) and np.array_equal(step,np.argmin(trace,axis=0)) and np.array_equal(best,trace[step,np.arange(825)])
   obs=risk(a['calibration_prefix'][ci,oi],calib_truth[:,origin:origin+3]);penalty=.5*np.sum((state.astype(np.float64)-mu)**2/variance,axis=1);obj=obs+(0. if coefficient is None else coefficient)*scale*penalty;objective_errors.append(float(np.max(np.abs(obj-best))));prefix_errors.append(float(np.max(np.abs(risk(images[:,:3],calib_truth[:,origin:origin+3])-obs))))
   if coefficient is None:assert np.array_equal(state,mu) and np.count_nonzero(step)==0
  print(json.dumps(dict(stage='calibration_bank_rescored',candidate=ci,max_error=max(metric_errors))),flush=True)
 assert max(objective_errors)<1e-5 and max(prefix_errors)<1e-5 and max(metric_errors)<1e-9
 env_scores=calib_risks.reshape(8,5,75,11).mean(axis=(1,3));chosen=np.argmin(env_scores,axis=0);assert chosen.tolist()==selection['chosen_candidate_indices']==a['chosen_candidate_indices'].tolist();assert [candidates[i] for i in chosen]==selection['chosen_coefficients'];assert np.max(np.abs(env_scores-selection['environment_scores']))<1e-9
 assert np.array_equal(a['query_codes'],a['full_environment_codes'][envs])
 oldr=read(ROOT/'rolling_origin_audit_v1/RESULTS_V1.json');oldpath=ART.parent/'rolling_origin_audit_v1/arrays.npz';assert sha_file(oldpath)==oldr['payload_sha256'];old=np.load(oldpath);query_rows=[];replay_errors=[]
 for path in ['lambda0','selected']:
  ids=np.full(300,7) if path=='lambda0' else chosen[envs];coeff=np.array([0. if candidates[i] is None else candidates[i] for i in ids]);active=ids!=0
  for oi,origin in enumerate(origins):
   image=a['query_'+path+'_prediction'][oi];assert np.isfinite(image).all();risks=risk(image[:,3:],query_truth[:,origin+3:origin+20]);metric_errors.append(float(np.max(np.abs(risks-a[path+'_risks_origin'+str(origin)]))));checks+=300
   mu=a['query_'+path+'_mu'][oi];lv=a['query_'+path+'_logvar'][oi];state=a['query_'+path+'_state'][oi];variance=a['query_'+path+'_variance'][oi];scale=a['query_'+path+'_scale'][oi];assert np.allclose(variance,np.maximum(np.exp(lv.astype(np.float64)),1e-8),atol=1e-10,rtol=1e-6)
   trace=a['query_'+path+'_trace'][oi];step=a['query_'+path+'_step'][oi];assert trace.shape==(61,300) and np.array_equal(step,np.argmin(trace,axis=0));obj=risk(a['query_'+path+'_prefix'][oi],query_truth[:,origin:origin+3])+coeff*scale*.5*np.sum((state.astype(np.float64)-mu)**2/variance,axis=1);objective_errors.append(float(np.max(np.abs(obj-trace[step,np.arange(300)]))));assert np.array_equal(state[~active],mu[~active])
   base=old[f'metric_reencode_absolute_origin{origin}'];row=dict(path=path,origin=origin,future_mse=float(risks.mean()),worse_than_no_update=int(np.sum(risks>base)),mean_positive_regret=float(np.maximum(risks-base,0).mean()),max_positive_regret=float(np.maximum(risks-base,0).max()));stored=next(x for x in r['query_results'] if x['path']==path and x['origin']==origin)
   for key in ['future_mse','worse_than_no_update','mean_positive_regret','max_positive_regret']:assert abs(row[key]-stored[key])<1e-9
   query_rows.append(row)
   if path=='lambda0':replay_errors.append(float(np.max(np.abs(image-old['fit_absolute'][oi]))))
 assert max(replay_errors)<1e-4 and max(objective_errors)<1e-5 and max(metric_errors)<1e-9
 # Fresh original context embeddings and actual forward replay, beyond metric recalc.
 torch.backends.cudnn.deterministic=True;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;sys.path.insert(0,str(SRC/'pendulum/meta_ilad'));from model import iLaD
 ckpt=SRC/'pendulum/checkpoints/meta_ilad_s1000.pt';assert sha_file(ckpt)==plan['checkpoint_sha256'];m=iLaD('cuda').cuda();m.load_state_dict(torch.load(ckpt,weights_only=True,map_location='cuda'));m.eval();grid=torch.arange(40,device='cuda',dtype=torch.float32)/10;embedding_errors=[];native_replays=[]
 with torch.no_grad():
  for env in range(75):
   actual=m.encoder.embed(raw['yt'][contexts[env],:40].cuda()).cpu().numpy();embedding_errors.append(float(np.max(np.abs(actual-a['support_embeddings'][env]))))
 assert max(embedding_errors)<1e-4
 for label,origin,z,c,expected in [('calibration_zero',0,a['calibration_state'][7,0,:8],a['calibration_codes'][:8],pred[7,0,:8]),('calibration_regularized',10,a['calibration_state'][3,2,:8],a['calibration_codes'][:8],pred[3,2,:8]),('query_selected',0,a['query_selected_state'][0,:8],a['query_codes'][:8],a['query_selected_prediction'][0,:8]),('query_selected',20,a['query_selected_state'][4,:8],a['query_codes'][:8],a['query_selected_prediction'][4,:8])]:
  with torch.no_grad():actual=m.decode(m.dynamics(torch.from_numpy(z.copy()).cuda(),torch.from_numpy(c.copy()).cuda(),compute_logp=False,t=grid[origin:origin+20])).cpu().numpy()
  err=float(np.max(np.abs(actual-expected)));assert err<1e-4,(label,origin,err);native_replays.append(dict(path=label,origin=origin,max_pixel_error=err))
 assert r['model_state_before']==r['model_state_after'];assert r['optimizer_calls']==2700 and r['new_network_weight_training_fits']==0 and not selection['query_future_metrics_used'];assert r['additional_observation_episodes']==0
 result=dict(status='FULL_CONTEXT_ONLY_STATE_REGULARIZATION_METRICS_AND_SELECTION_VERIFIED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),image_metric_checks=checks,max_image_metric_error=max(metric_errors),max_selected_objective_error=max(objective_errors),max_full_vs_prefix_error=max(prefix_errors),all_context_exclusions_exact=True,all75_environment_selections_reconstructed=True,selection_query_future_used=False,additional_observation_episodes=0,chosen_candidate_counts=np.bincount(chosen,minlength=8).tolist(),candidates=candidates,max_head_reconstruction_error=max(head_errors),max_fresh_support_embedding_error=max(embedding_errors),lambda0_replay_max_error=max(replay_errors),fresh_native_forward_replays=native_replays,query_results=query_rows,optimizer_calls=2700,new_network_weight_training_fits=0,novel_direction_accepted=False,local_payload_sync=False)
 dump('VERIFICATION_V1.json',result);print(json.dumps(result),flush=True)
except Exception as exc:
 dump('FAILURE_V1.json',dict(exception_type=type(exc).__name__,exception=str(exc),traceback=traceback.format_exc()));raise
