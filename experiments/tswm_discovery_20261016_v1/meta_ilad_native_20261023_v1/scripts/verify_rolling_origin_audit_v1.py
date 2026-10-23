"""Independently reconstruct selection, source timing, prefix choices and image risks."""
import pathlib,sys,json,hashlib,datetime,traceback
import numpy as np,torch
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'rolling_origin_verification_v1';OUT.mkdir(exist_ok=False)
ART=pathlib.Path(r'G:\ICLR Omni-eeg cache\tswm_discovery_20261016_v1\artifacts\meta_ilad_native_20261023_v1\rolling_origin_audit_v1\arrays.npz')
SRC=pathlib.Path(r'G:\ICLR Omni-eeg cache\tswm_discovery_20261016_v1\caches\meta_ilad_native_20261023_v1\source')
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
def sha_file(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(n,x):
 with (OUT/n).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def score(p,t):
 d=np.asarray(p,dtype=np.float64)-t;return np.einsum('nlhw,nlhw->n',d,d)/d.shape[1]
try:
 torch.set_num_threads(1);r=read(ROOT/'rolling_origin_audit_v1/RESULTS_V1.json');plan=read(ROOT/'rolling_origin_audit_v1/PLAN_V1.json');exit_receipt=read(ROOT/'rolling_origin_execution_v1/exit.json')
 assert exit_receipt['exit_code']==0 and exit_receipt['log_sha256']==sha_file(ROOT/'rolling_origin_execution_v1/stdout.log')
 assert plan['protocol_sha256']==sha_file(ROOT/'ROLLING_ORIGIN_AUDIT_PROTOCOL_V1.md') and sha_file(ART)==r['payload_sha256']
 for n,h in plan['source_hashes'].items():assert sha_file(SRC/'pendulum/meta_ilad'/n)==h
 a=np.load(ART);data=next(x for x in read(ROOT/'pendulum_data_generation_v1/RESULTS_V1.json')['sets'] if x['split']=='test');assert sha_file(data['path'])==plan['data_sha256'];raw=torch.load(data['path'],weights_only=True,mmap=True,map_location='cpu');labels=raw['label'].reshape(-1).numpy();assert np.array_equal(labels,np.tile(np.arange(75),40))
 # Recreate native bank with its own documented legacy RNG, then query RNG separately.
 native=np.random.RandomState(42);banks={}
 for env in range(75):
  indexes=np.flatnonzero(labels==env);order=np.arange(40);native.shuffle(order);banks[env]=indexes[np.sort(order[:13])]
 rng=np.random.default_rng(202610234);q=[];contexts=[]
 for env in range(75):
  assert plan['native_support13'][str(env)]==banks[env].tolist();contexts.append(banks[env][:11]);allowed=np.setdiff1d(np.flatnonzero(labels==env),banks[env]);q.extend(sorted(rng.choice(allowed,4,replace=False).tolist()))
 q=np.asarray(q);contexts=np.asarray(contexts);assert np.array_equal(q,a['queries']) and q.tolist()==plan['queries'];assert np.array_equal(contexts,a['contexts']) and contexts.tolist()==plan['contexts'];assert not set(q).intersection(contexts.reshape(-1));assert np.array_equal(labels[q],a['environment_ids']) and np.array_equal(labels[q],np.repeat(np.arange(75),4))
 origins=[0,5,10,15,20];assert a['origins'].tolist()==plan['origins']==origins;assert plan['relative_future_frames']==list(range(3,20)) and max(origins)+19==39
 truth=raw['yt'][q,:40].numpy().astype(np.float64);paths=['reencode_absolute','fit_absolute','carry_from_zero','reencode_reset_negative_control'];metric_errors=[];selection_errors=[];rows=[];checks=0;restart_errors=[]
 for group,origin in enumerate(origins):
  true=truth[:,origin:origin+20];means={}
  for name in paths:
   image=a[name][group];assert image.shape==(300,20,32,32) and np.isfinite(image).all();risk=score(image[:,3:],true[:,3:]);stored=a[f'metric_{name}_origin{origin}'];metric_errors.append(float(np.max(np.abs(risk-stored))));checks+=300;means[name]=float(risk.mean());metric_errors.append(abs(means[name]-r['origins'][group]['means'][name]))
   if name=='carry_from_zero':assert np.array_equal(image,a['carry_images'][:,origin:origin+20])
  persistence=score(np.repeat(true[:,2:3],17,1),true[:,3:]);metric_errors.append(float(np.max(np.abs(persistence-a[f'metric_persistence_origin{origin}']))));checks+=300;means['persistence']=float(persistence.mean());metric_errors.append(abs(means['persistence']-r['origins'][group]['means']['persistence']))
  trace=a['prefix_loss_traces'][group];step=a['prefix_selected_steps'][group];assert trace.shape==(61,300) and np.array_equal(step,np.argmin(trace,axis=0));best=trace[step,np.arange(300)];actual=score(a['selected_prefix_predictions'][group],true[:,:3]);selection_errors.append(float(np.max(np.abs(actual-best))))
  restart=float(np.max(np.abs(a[f'restart_origin{origin}']-a['carry_images'][:,origin:origin+20])));restart_errors.append(restart);assert restart==r['origins'][group]['restart_pixel_max_error'];rows.append(dict(origin=origin,absolute_time=origin/10,means=means,restart_pixel_max_error=restart))
 assert max(metric_errors)<1e-9 and max(selection_errors)<1e-5
 identity=float(np.max(np.abs(a['reencode_absolute'][0]-a['reencode_reset_negative_control'][0])));assert identity==r['identity_reset_origin0_error'];qualified=bool(max(restart_errors)<1e-4 and identity<1e-4);assert qualified==r['restart_qualification_passed']
 assert r['model_state_before']==r['model_state_after'] and r['state_optimizer_calls']==300 and r['new_network_weight_training_fits']==0
 # New original-source forward replay, with batch-size variation qualified separately.
 torch.backends.cudnn.deterministic=True;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;sys.path.insert(0,str(SRC/'pendulum/meta_ilad'));from model import iLaD
 ckpt=SRC/'pendulum/checkpoints/meta_ilad_s1000.pt';assert sha_file(ckpt)==plan['checkpoint_sha256'];m=iLaD('cuda').cuda();m.load_state_dict(torch.load(ckpt,weights_only=True,map_location='cuda'));m.eval();grid=torch.arange(40,device='cuda',dtype=torch.float32)/10;replays=[]
 for group,fit,reset in [(0,False,False),(2,False,False),(2,True,False),(2,False,True),(4,True,False)]:
  origin=origins[group];key='fit_absolute' if fit else ('reencode_reset_negative_control' if reset else 'reencode_absolute');z=torch.from_numpy(a['fitted_states' if fit else 'initial_states'][group,:8].copy()).cuda();c=torch.from_numpy(a['codes'][:8].copy()).cuda();times=grid[:20] if reset else grid[origin:origin+20]
  with torch.no_grad():pred=m.decode(m.dynamics(z,c,compute_logp=False,t=times)).cpu().numpy()
  err=float(np.max(np.abs(pred-a[key][group,:8])));assert err<1e-4,(origin,key,err);replays.append(dict(origin=origin,path=key,max_pixel_error=err))
 result=dict(status='ROLLING_ORIGIN_SELECTION_IMAGE_METRICS_AND_NATIVE_REPLAY_VERIFIED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),query_metric_checks=checks,max_image_metric_error=max(metric_errors),max_prefix_selection_error=max(selection_errors),source_support_and_query_rng_exact=True,all_queries_excluded_from_context=True,all_future_times_within_original40=True,all_selection_prefix_only=True,origins=rows,identity_reset_origin0_error=identity,max_carried_restart_error=max(restart_errors),carried_restart_qualification_passed=qualified,qualification_tolerance_unchanged=1e-4,fresh_native_forward_replays=replays,state_optimizer_calls=300,new_network_weight_training_fits=0,accepted_novel_directions=0,local_payload_sync=False)
 dump('VERIFICATION_V1.json',result);print(json.dumps(result),flush=True)
except Exception as exc:
 dump('FAILURE_V1.json',dict(exception_type=type(exc).__name__,exception=str(exc),traceback=traceback.format_exc()));raise
