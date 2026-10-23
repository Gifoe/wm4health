"""Native fixed-weight rolling-origin audit; preserve absolute external forcing time."""
import pathlib,sys,json,hashlib,datetime,time,traceback
import numpy as np,torch
ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'rolling_origin_audit_v1';OUT.mkdir(exist_ok=False)
ART=pathlib.Path(r'G:\ICLR Omni-eeg cache\tswm_discovery_20261016_v1\artifacts\meta_ilad_native_20261023_v1\rolling_origin_audit_v1');ART.mkdir(exist_ok=False)
SRC=pathlib.Path(r'G:\ICLR Omni-eeg cache\tswm_discovery_20261016_v1\caches\meta_ilad_native_20261023_v1\source')
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
 d=p.astype(np.float64)-t.astype(np.float64);return np.einsum('nlhw,nlhw->n',d,d)/d.shape[1]
try:
 torch.set_num_threads(1);torch.manual_seed(1000);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
 sys.path.insert(0,str(SRC/'pendulum/meta_ilad'));from model import iLaD
 from dataloader import MyDataset
 data=next(x for x in read(ROOT/'pendulum_data_generation_v1/RESULTS_V1.json')['sets'] if x['split']=='test');assert sha_file(data['path'])==data['file_sha256']
 ds=MyDataset(data['path'],12,42,'cuda');assert len(ds)==3000 and np.array_equal(ds.labels,np.tile(np.arange(75),40))
 m=iLaD('cuda').cuda();ckpt=SRC/'pendulum/checkpoints/meta_ilad_s1000.pt';assert sha_file(ckpt)==read(ROOT/'context_transplant_audit_v1/PLAN_V1.json')['checkpoint_sha256'];m.load_state_dict(torch.load(ckpt,weights_only=True,map_location='cuda'));m.eval()
 for p in m.parameters():p.requires_grad_(False)
 before=digest(m);assert before==read(ROOT/'joint_context_refinement_v1/RESULTS_V1.json')['model_state_after']
 rng=np.random.default_rng(202610234);queries=[];contexts=[];environments=[]
 for env in range(75):
  bank=np.asarray(ds.spt_idx[env]);contexts.append(bank[:11]);allowed=np.setdiff1d(np.arange(env,3000,75),bank);selected=np.sort(rng.choice(allowed,4,replace=False));queries.extend(selected.tolist());environments.extend([env]*4)
 queries=np.array(queries);contexts=np.stack(contexts);environments=np.array(environments);assert len(queries)==300 and not set(queries).intersection(contexts.reshape(-1))
 origins=np.array([0,5,10,15,20]);grid=torch.arange(40,device='cuda',dtype=torch.float32)/10
 dump('PLAN_V1.json',dict(protocol_sha256=sha_file(ROOT/'ROLLING_ORIGIN_AUDIT_PROTOCOL_V1.md'),data_sha256=data['file_sha256'],checkpoint_sha256=sha_file(ckpt),source_revision=read(ROOT/'pendulum_data_generation_v1/RESULTS_V1.json')['source_revision'],source_hashes={n:sha_file(SRC/'pendulum/meta_ilad'/n) for n in ['model.py','dynamics.py','vae.py','dataloader.py']},queries=queries.tolist(),environment_ids=environments.tolist(),native_support13={str(env):ds.spt_idx[env].tolist() for env in range(75)},contexts=contexts.tolist(),origins=origins.tolist(),relative_future_frames=list(range(3,20)),seed=202610234,prefix_optimizer_steps_per_origin=60,new_weight_training_fits=0))
 with torch.no_grad():
  codes=[]
  for indexes in contexts:
   h=m.encoder.embed(ds.images[torch.from_numpy(indexes).cuda(),:40]).mean(0,keepdim=True);codes.append(m.encoder.fc_mu(h)[0])
  env_codes=torch.stack(codes);c=env_codes[torch.from_numpy(environments).cuda()];xs=ds.images[torch.from_numpy(queries).cuda(),:40];z_start,_=m.vae.encode(xs[:,:3].contiguous());carry_z=m.dynamics(z_start,c,compute_logp=False,t=grid);carry_images=m.decode(carry_z)
 def rollout(z,t):return m.decode(m.dynamics(z,c,compute_logp=False,t=t))
 arrays=dict(queries=queries,environment_ids=environments,contexts=contexts,origins=origins,environment_codes=env_codes.cpu().numpy(),codes=c.cpu().numpy(),carry_latents=carry_z.cpu().numpy(),carry_images=carry_images.cpu().numpy());metrics=[];restart_errors=[];paths=['reencode_absolute','fit_absolute','carry_from_zero','reencode_reset_negative_control'];preds={p:[] for p in paths};states=[];fitted=[];traces=[];selections=[];prefix_predictions=[]
 for origin in origins.tolist():
  start=time.monotonic();times=grid[origin:origin+20];hist=xs[:,origin:origin+3].contiguous();truth=xs[:,origin:origin+20].cpu().numpy()
  with torch.no_grad():
   z,_=m.vae.encode(hist);p=rollout(z,times);wrong=rollout(z,grid[:20]);initial_prefix=rollout(z,times[:3]);best_loss=(initial_prefix-hist).square().sum((2,3)).mean(1);best=z.clone();best_step=torch.zeros(300,device='cuda',dtype=torch.int64)
   restart=rollout(carry_z[:,origin],times);restart_error=float((restart-carry_images[:,origin:origin+20]).abs().max());restart_errors.append(restart_error)
  state=torch.nn.Parameter(z.clone());optimizer=torch.optim.Adam([state],lr=.01);trace=[best_loss.cpu().numpy()]
  for step in range(1,61):
   optimizer.zero_grad(set_to_none=True);loss=(rollout(state,times[:3])-hist).square().sum((2,3)).mean(1);loss.mean().backward();torch.nn.utils.clip_grad_norm_([state],10.);optimizer.step()
   with torch.no_grad():
    current=(rollout(state,times[:3])-hist).square().sum((2,3)).mean(1);assert torch.isfinite(current).all();improved=current<best_loss;best[improved]=state.detach()[improved];best_loss[improved]=current[improved];best_step[improved]=step;trace.append(current.cpu().numpy())
  with torch.no_grad():fit=rollout(best,times);prefix=rollout(best,times[:3])
  values=[p.cpu().numpy(),fit.cpu().numpy(),carry_images[:,origin:origin+20].cpu().numpy(),wrong.cpu().numpy()]
  origin_metrics={}
  for name,prediction in zip(paths,values):
   assert np.isfinite(prediction).all();preds[name].append(prediction);score=risk(prediction[:,3:],truth[:,3:]);origin_metrics[name]=float(score.mean());arrays[f'metric_{name}_origin{origin}']=score
  persistence=risk(np.repeat(truth[:,2:3],17,1),truth[:,3:]);arrays[f'metric_persistence_origin{origin}']=persistence;origin_metrics['persistence']=float(persistence.mean())
  if origin==0:identity_error=float(np.max(np.abs(values[0]-values[3])))
  arrays[f'restart_origin{origin}']=restart.cpu().numpy();states.append(z.cpu().numpy());fitted.append(best.cpu().numpy());traces.append(np.stack(trace));selections.append(best_step.cpu().numpy());prefix_predictions.append(prefix.cpu().numpy());metrics.append(dict(origin=origin,absolute_time=origin/10,future_frames=17,means=origin_metrics,restart_pixel_max_error=restart_error));event('origin_complete',**metrics[-1],seconds=time.monotonic()-start)
 for name,values in preds.items():arrays[name]=np.stack(values)
 arrays.update(initial_states=np.stack(states),fitted_states=np.stack(fitted),prefix_loss_traces=np.stack(traces),prefix_selected_steps=np.stack(selections),selected_prefix_predictions=np.stack(prefix_predictions))
 assert digest(m)==before
 payload=ART/'arrays.npz'
 with payload.open('xb') as f:np.savez_compressed(f,**arrays)
 result=dict(status='ROLLING_ORIGIN_NATIVE_STATE_AUDIT_COMPLETE',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),origins=metrics,query_sequences=300,environments=75,repeated_query_origins=1500,identity_reset_origin0_error=identity_error,max_carried_restart_error=max(restart_errors),restart_qualification_passed=bool(max(restart_errors)<1e-4 and identity_error<1e-4),qualification_tolerance_unchanged=1e-4,context_code_fits=0,state_optimizer_calls=300,new_network_weight_training_fits=0,model_state_before=before,model_state_after=digest(m),payload_sha256=sha_file(payload),payload_bytes=payload.stat().st_size,independent_verification_pending=True,accepted_novel_directions=0,local_payload_sync=False)
 dump('RESULTS_V1.json',result);event('audit_terminal',status=result['status'],qualification=result['restart_qualification_passed'],max_carried_restart_error=max(restart_errors))
except Exception as exc:
 dump('FAILURE_V1.json',dict(exception_type=type(exc).__name__,exception=str(exc),traceback=traceback.format_exc()));raise
