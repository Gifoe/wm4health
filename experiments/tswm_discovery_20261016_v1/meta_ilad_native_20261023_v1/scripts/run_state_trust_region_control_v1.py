"""Ordinary proximal state fitting; all hyperparameter selection is context-only."""
import pathlib,sys,json,hashlib,datetime,time,traceback
import numpy as np,torch
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'state_trust_region_control_v1';OUT.mkdir(exist_ok=False)
ART=pathlib.Path(r'G:\ICLR Omni-eeg cache\tswm_discovery_20261016_v1\artifacts\meta_ilad_native_20261023_v1\state_trust_region_control_v1');ART.mkdir(exist_ok=False)
SRC=pathlib.Path(r'G:\ICLR Omni-eeg cache\tswm_discovery_20261016_v1\caches\meta_ilad_native_20261023_v1\source')
CANDIDATES=[None,10.,1.,.1,.01,.001,.0001,0.];ORIGINS=[0,5,10,15,20]
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
def sha_file(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(n,x):
 with (OUT/n).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def event(stage,**kw):
 x=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),stage=stage,**kw)
 with (OUT/'trace.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(x)+'\n')
 print(json.dumps(x),flush=True)
def digest(m):
 h=hashlib.sha256()
 for k,v in sorted(m.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def risk(p,t):
 d=np.asarray(p,dtype=np.float64)-np.asarray(t,dtype=np.float64);return np.einsum('nlhw,nlhw->n',d,d)/d.shape[1]
def save_npz(p,arrays):
 with p.open('xb') as f:np.savez_compressed(f,**arrays)
try:
 assert read(ROOT/'rolling_origin_verification_v1/VERIFICATION_V1.json')['carried_restart_qualification_passed']
 oldplan=read(ROOT/'rolling_origin_audit_v1/PLAN_V1.json');oldr=read(ROOT/'rolling_origin_audit_v1/RESULTS_V1.json')
 torch.set_num_threads(1);torch.manual_seed(1000);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
 sys.path.insert(0,str(SRC/'pendulum/meta_ilad'));from model import iLaD
 from dataloader import MyDataset
 data=next(x for x in read(ROOT/'pendulum_data_generation_v1/RESULTS_V1.json')['sets'] if x['split']=='test');assert sha_file(data['path'])==oldplan['data_sha256'];ds=MyDataset(data['path'],12,42,'cuda')
 m=iLaD('cuda').cuda();ckpt=SRC/'pendulum/checkpoints/meta_ilad_s1000.pt';assert sha_file(ckpt)==oldplan['checkpoint_sha256'];m.load_state_dict(torch.load(ckpt,weights_only=True,map_location='cuda'));m.eval()
 for p in m.parameters():p.requires_grad_(False)
 before=digest(m);assert before==oldr['model_state_after'];contexts=np.asarray(oldplan['contexts']);queries=np.asarray(oldplan['queries']);envs=np.asarray(oldplan['environment_ids']);assert contexts.shape==(75,11) and not set(queries).intersection(contexts.reshape(-1))
 grid=torch.arange(40,device='cuda',dtype=torch.float32)/10
 with torch.no_grad():
  support_h=[];full_codes=[];loo_codes=[]
  for bank in contexts:
   h=m.encoder.embed(ds.images[torch.from_numpy(bank).cuda(),:40]);support_h.append(h);full_codes.append(m.encoder.fc_mu(h.mean(0,keepdim=True))[0])
   for leave in range(11):loo_codes.append(m.encoder.fc_mu(h[torch.arange(11,device='cuda')!=leave].mean(0,keepdim=True))[0])
  full_codes=torch.stack(full_codes);calib_codes=torch.stack(loo_codes);calib_x=ds.images[torch.from_numpy(contexts.reshape(-1)).cuda(),:40]
 dump('PLAN_V1.json',dict(protocol_sha256=sha_file(ROOT/'STATE_TRUST_REGION_CONTROL_PROTOCOL_V1.md'),data_sha256=data['file_sha256'],checkpoint_sha256=sha_file(ckpt),old_plan_sha256=sha_file(ROOT/'rolling_origin_audit_v1/PLAN_V1.json'),queries=queries.tolist(),environment_ids=envs.tolist(),contexts=contexts.tolist(),calibration_indices=contexts.reshape(-1).tolist(),candidates=CANDIDATES,origins=ORIGINS,variance_floor=1e-8,scale_floor=1e-6,steps=60,learning_rate=.01,gradient_clip=10.,source_revision=oldplan['source_revision'],selection_scope='one candidate per environment from own 11 completed contexts only',new_weight_training_fits=0))
 def fit(hist,c,times,coeff,active):
  n=len(hist)
  def rollout(z,t):return m.decode(m.dynamics(z,c,compute_logp=False,t=t))
  with torch.no_grad():
   mu,lv=m.vae.encode(hist);var=torch.exp(lv).clamp_min(1e-8);initial=rollout(mu,times[:3]);initial_loss=(initial-hist).square().sum((2,3)).mean(1);scale=initial_loss.clamp_min(1e-6);best=mu.clone();best_step=torch.zeros(n,device='cuda',dtype=torch.int64);best_loss=initial_loss.clone()
  def objectives(z):
   prefix=rollout(z,times[:3]);obs=(prefix-hist).square().sum((2,3)).mean(1);pen=.5*((z-mu).square()/var).sum(1);return obs+coeff*scale*pen
  trace=[best_loss.cpu().numpy()]
  if bool(active.any()):
   state=torch.nn.Parameter(mu.clone());opt=torch.optim.Adam([state],lr=.01)
   for step in range(1,61):
    opt.zero_grad(set_to_none=True);loss=objectives(state);(loss*active).mean().backward();torch.nn.utils.clip_grad_norm_([state],10.);opt.step()
    with torch.no_grad():
     state[~active]=mu[~active];current=objectives(state);assert torch.isfinite(current).all();improved=current<best_loss;best[improved]=state.detach()[improved];best_step[improved]=step;best_loss[improved]=current[improved];trace.append(current.cpu().numpy())
  else:trace=[initial_loss.cpu().numpy() for _ in range(61)]
  with torch.no_grad():prediction=rollout(best,times);prefix=rollout(best,times[:3])
  return dict(prediction=prediction.cpu().numpy(),prefix=prefix.cpu().numpy(),mu=mu.cpu().numpy(),logvar=lv.cpu().numpy(),variance=var.cpu().numpy(),scale=scale.cpu().numpy(),state=best.cpu().numpy(),trace=np.stack(trace),step=best_step.cpu().numpy(),best_loss=best_loss.cpu().numpy())
 meta=dict(support_embeddings=torch.stack(support_h).cpu().numpy(),mu_head_weight=m.encoder.fc_mu.weight.detach().cpu().numpy(),mu_head_bias=m.encoder.fc_mu.bias.detach().cpu().numpy(),contexts=contexts,queries=queries,environment_ids=envs,calibration_indices=contexts.reshape(-1),calibration_codes=calib_codes.cpu().numpy(),full_environment_codes=full_codes.cpu().numpy());fits={};calib_risks=np.empty((8,5,825),dtype=np.float64)
 cp=ART/'calibration_predictions.npy';calib_store=np.lib.format.open_memmap(cp,mode='w+',dtype=np.float32,shape=(8,5,825,20,32,32));optimizer_calls=0
 for ci,coefficient in enumerate(CANDIDATES):
  started=time.monotonic();origin_records=[]
  for oi,origin in enumerate(ORIGINS):
   coeff=torch.full((825,),0. if coefficient is None else coefficient,device='cuda');active=torch.full((825,),coefficient is not None,device='cuda',dtype=torch.bool);f=fit(calib_x[:,origin:origin+3].contiguous(),calib_codes,grid[origin:origin+20],coeff,active);optimizer_calls+=60 if coefficient is not None else 0
   calib_store[ci,oi]=f.pop('prediction');calib_risks[ci,oi]=risk(calib_store[ci,oi,:,3:],calib_x[:,origin+3:origin+20].cpu().numpy());origin_records.append({k:x for k,x in f.items()})
  fits[ci]=origin_records;event('calibration_candidate_complete',candidate=ci,coefficient=coefficient,future_mse=float(calib_risks[ci].mean()),seconds=time.monotonic()-started)
 calib_store.flush();del calib_store
 for key in fits[0][0]:meta['calibration_'+key]=np.stack([[fits[ci][oi][key] for oi in range(5)] for ci in range(8)])
 meta['calibration_risks']=calib_risks;calib_scores=calib_risks.reshape(8,5,75,11).mean((1,3));chosen=np.argmin(calib_scores,axis=0);meta['calibration_environment_scores']=calib_scores;meta['chosen_candidate_indices']=chosen
 dump('SELECTION_V1.json',dict(candidates=CANDIDATES,environment_scores=calib_scores.tolist(),chosen_candidate_indices=chosen.tolist(),chosen_coefficients=[CANDIDATES[x] for x in chosen],query_future_metrics_used=False,external_context_episodes_added=0,tie_break='earlier candidate index',utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
 selection_sha=sha_file(OUT/'SELECTION_V1.json');event('context_only_selection_persisted',selection_sha256=selection_sha,candidate_counts=np.bincount(chosen,minlength=8).tolist())
 # Only now inspect original-query future predictions or compute new query scores.
 oldpath=ART.parent/'rolling_origin_audit_v1/arrays.npz';assert sha_file(oldpath)==oldr['payload_sha256'];old=np.load(oldpath);codes=full_codes[torch.from_numpy(envs).cuda()];xs=ds.images[torch.from_numpy(queries).cuda(),:40];query_fits={};scores=[];qualification=[]
 for path in ['lambda0','selected']:
  records=[]
  for oi,origin in enumerate(ORIGINS):
   ids=np.full(300,7) if path=='lambda0' else chosen[envs];coeff=torch.tensor([0. if CANDIDATES[x] is None else CANDIDATES[x] for x in ids],device='cuda');active=torch.tensor(ids!=0,device='cuda');f=fit(xs[:,origin:origin+3].contiguous(),codes,grid[origin:origin+20],coeff,active);optimizer_calls+=60
   risk_values=risk(f['prediction'][:,3:],xs[:,origin+3:origin+20].cpu().numpy());meta[path+'_risks_origin'+str(origin)]=risk_values;records.append(f)
   if path=='lambda0':
    err=float(np.max(np.abs(f['prediction']-old['fit_absolute'][oi])));zerr=float(np.max(np.abs(f['mu']-old['initial_states'][oi])));cerr=float(np.max(np.abs(codes.cpu().numpy()-old['codes'])));qualification.append(dict(origin=origin,max_pixel_error=err,max_state_error=zerr,max_code_error=cerr));assert err<1e-4 and zerr<1e-5 and cerr<1e-5
   base=old[f'metric_reencode_absolute_origin{origin}'];scores.append(dict(path=path,origin=origin,future_mse=float(risk_values.mean()),worse_than_no_update=int(np.count_nonzero(risk_values>base)),mean_positive_regret=float(np.maximum(risk_values-base,0).mean()),max_positive_regret=float(np.maximum(risk_values-base,0).max())))
  query_fits[path]=records
 for path,records in query_fits.items():
  for key in records[0]:meta['query_'+path+'_'+key]=np.stack([x[key] for x in records])
 meta['query_codes']=codes.cpu().numpy();assert sha_file(OUT/'SELECTION_V1.json')==selection_sha and digest(m)==before;mp=ART/'metadata_and_query_predictions.npz';save_npz(mp,meta)
 result=dict(status='CONTEXT_ONLY_STATE_TRUST_REGION_CONTROL_COMPLETE',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),selection_sha256=selection_sha,chosen_candidate_counts=np.bincount(chosen,minlength=8).tolist(),candidates=CANDIDATES,calibration_candidate_environment_scores=calib_scores.tolist(),query_results=scores,lambda0_replay_qualification=qualification,all_query_parameters_selected_from_context_only=True,additional_observation_episodes=0,calibration_leave_one_out_code_episodes=10,deployment_code_episodes=11,optimizer_calls=optimizer_calls,calibration_optimizer_calls=2100,query_optimizer_calls=600,new_network_weight_training_fits=0,model_state_before=before,model_state_after=digest(m),payloads=[dict(name=p.name,bytes=p.stat().st_size,sha256=sha_file(p)) for p in [cp,mp]],independent_verification_pending=True,accepted_novel_directions=0,local_payload_sync=False)
 dump('RESULTS_V1.json',result);event('control_complete',candidate_counts=result['chosen_candidate_counts'],query_results=scores,optimizer_calls=optimizer_calls)
except Exception as exc:
 dump('FAILURE_V1.json',dict(exception_type=type(exc).__name__,exception=str(exc),traceback=traceback.format_exc()));raise
