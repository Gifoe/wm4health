"""Real native-core factual transfer; frozen protocol, no intervention queries."""
import os
os.environ.update(JAX_PLATFORMS='cpu',OMP_NUM_THREADS='4',TF_NUM_INTRAOP_THREADS='4',TF_NUM_INTEROP_THREADS='1',XLA_FLAGS='--xla_cpu_multi_thread_eigen=false',WANDB_MODE='disabled')
import sys,pathlib,json,datetime,time,traceback,faulthandler,pickle
faulthandler.enable(all_threads=True)
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from native_core_loader_v1 import ROOT,read,sha,qualify_source,load_core
import numpy as np,jax,jax.numpy as jnp
OUT=ROOT/'factual_fit_recovery_v1';OUT.mkdir(exist_ok=False)
SEED=202610231;TOTAL=2000;CHECKS=[0,500,1000,2000]
def dump(p,x):
 with pathlib.Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def event(stage,**kw):
 x=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),stage=stage,**kw)
 with (OUT/'trace.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(x,allow_nan=False)+'\n')
 print(json.dumps(x,allow_nan=False),flush=True)
def finite(tree):
 for leaf in jax.tree_util.tree_leaves(tree):
  if hasattr(leaf,'dtype'):assert np.isfinite(np.asarray(leaf)).all(),('nonfinite',getattr(leaf,'shape',None))
def model_vars(v):return {k:x for k,x in v.items() if 'model_opt' not in k}
def make_data(x,a):
 b,t=x.shape[:2];return dict(obs=jnp.asarray(x),action=jnp.asarray(a),reward=jnp.zeros((b,t)),cont=jnp.ones((b,t)),is_first=jnp.zeros((b,t),bool).at[:,0].set(True),is_last=jnp.zeros((b,t),bool),is_terminal=jnp.zeros((b,t),bool))
try:
 proof=read(ROOT/'native_core_profile_diagnostic_v1/RESULTS_V1.json');assert proof['qualified_optimizer_updates']==3 and proof['third_call_retraced'] is False
 wm,nj,config,source=load_core();dump(OUT/'CONFIG_V1.json',dict(config));storage=read(ROOT.parent/'STORAGE_LOCATION_V1.json');ART=pathlib.Path(storage['checkpoint_root'])/ROOT.name/'factual_fit_recovery_v1';ART.mkdir(parents=True,exist_ok=False)
 payload=pathlib.Path(storage['artifact_root'])/ROOT.name/'factual_fit_recovery_v1';payload.mkdir(parents=True,exist_ok=False)
 manifest=read(ROOT.parent/'cycle88/outputs/PLAN_DATA88_V1.json');datasets={};records={}
 for split in ['train','val']:
  row=next(x for x in manifest['files'] if x['split']==split);assert sha(row['path'])==row['sha256'];records[split]=row
  with np.load(row['path'],allow_pickle=False) as z:datasets[split]={k:z[k].copy() for k in ['observations','actions','family']}
 lag=read(ROOT.parent/'cycle88/outputs/PLAN_FIT88_V1_lagged.json');references=lag['selected']['families'];dump(OUT/'MATRIX_V1.json',dict(protocol_sha256=sha(ROOT/'FACTUAL_FIT_PROTOCOL_V1.md'),source_revision=source['source_revision'],seed=SEED,updates=TOTAL,batch=1,length=64,validation_steps=CHECKS,monte_carlo_draws=4,data=records,ordinary_reference=references,scientific_fits=1,original_incomplete_optimizer_calls=500,replayed_prefix_calls=500,recovery_sha256=sha(ROOT/'FACTUAL_CHECKPOINT_RECOVERY_V1.md'),paper_reproduction=False,qualification_weights_reused=False,intervention_or_eval_accessed=False))
 dump(OUT/'ENVIRONMENT_V1.json',dict(python=sys.version,devices=[str(x) for x in jax.devices()],packages=__import__('importlib.metadata',fromlist=['distributions']) and {d.metadata['Name']:d.version for d in __import__('importlib.metadata',fromlist=['distributions']).distributions()}))
 rng=np.random.default_rng(SEED);train=datasets['train'];val=datasets['val'];history=val['observations'][:,48:48+65];act=val['actions'][:,48:48+96];truth=val['observations'][:,48+65:48+97];family=val['family']
 def predict(history,past_action,future_action):
  # Context uses64 past observation/action pairs; RSSM root includes65 observed states.
  embed=wm.encoder({'obs':history});ctx=wm.ctx_encoder({'obs':history[:,:64],'action':past_action,'embed':embed[:,:64] if 'embed' in config.ctx_encoder.inputs else None})
  ctx65=jnp.broadcast_to(ctx[:,:1],(1,65,ctx.shape[-1]));previous=jnp.concatenate([jnp.zeros_like(past_action[:,:1]),past_action],1);first=jnp.zeros((1,65),bool).at[:,0].set(True)
  post,_=wm.rssm.observe(embed,previous,first,dcontext=ctx65);root={k:v[:,-1] for k,v in post.items()};ctx32=jnp.broadcast_to(ctx[:,:1],(1,32,ctx.shape[-1]));prior=wm.rssm.imagine(future_action,root,dcontext=ctx32)
  pred=wm.heads['decoder']({**prior,'context':ctx32})['obs'].mode();recon=wm.heads['decoder']({**post,'context':ctx65})['obs'].mode()
  return dict(prediction=pred,reconstruction=recon,context=ctx)
 eval_fn=jax.jit(nj.pure(predict));init_fn=jax.jit(nj.pure(lambda d:wm.loss(d,wm.initial(1))));train_fn=jax.jit(nj.pure(lambda d:wm.train(d,wm.initial(1))))
 original=read(ROOT/'factual_fit_v1/VALIDATION_STEP0000_V1.json');assert sha(original['checkpoint_path'])==original['checkpoint_sha256'];assert read(ROOT/'factual_execution_v1/exit.json')['exit_code']==1
 with np.load(original['checkpoint_path'],allow_pickle=False) as z:variables={k:jnp.asarray(z[k]) for k in z.files}
 assert not any('model_opt' in k for k in variables);finite(variables);rng.bit_generator.state=original['numpy_rng_state']
 oldprogress={}
 for line in (ROOT/'factual_fit_v1/trace.jsonl').read_text().splitlines():
  x=json.loads(line)
  if x['stage']=='factual_train_progress':oldprogress[x['step']]=x['metrics']
 replay_comparisons=[];event('verified_initial_checkpoint_restored',checkpoint_sha256=original['checkpoint_sha256'],optimizer_updates=0,original_incomplete_optimizer_calls=500)
 def save_full_state(step):
  path=ART/f'state{step:04d}.pkl';state=dict(variables=jax.tree_util.tree_map(lambda x:np.asarray(x),variables),numpy_rng_state=rng.bit_generator.state,next_update_prng_seed=SEED+step+1,step=step,seed=SEED)
  with path.open('xb') as f:pickle.dump(state,f,protocol=5)
  digest=sha(path)
  with path.open('rb') as f:restored=pickle.load(f)
  assert sha(path)==digest
  before,bt=jax.tree_util.tree_flatten(state['variables']);after,at=jax.tree_util.tree_flatten(restored['variables']);assert bt==at and len(before)==len(after)
  assert all(x.dtype==y.dtype and x.shape==y.shape and np.array_equal(x,y) for x,y in zip(before,after));assert state['numpy_rng_state']==restored['numpy_rng_state']
  receipt=dict(step=step,full_state_path=str(path),full_state_sha256=digest,leaves=len(before),serialization_round_trip_exact=True,numpy_rng_state=rng.bit_generator.state,next_update_prng_seed=SEED+step+1)
  dump(OUT/f'STATE_STEP{step:04d}_V1.json',receipt);return receipt
 all_rows=[];best=float('inf');chosen=None;started=time.perf_counter();time_since=time.perf_counter()
 for step in range(TOTAL+1):
  if step in CHECKS:
   event('factual_validation_start',step=step);v=model_vars(variables);prediction=[];reconstruction=[];contexts=[];eval_start=time.perf_counter();replay_max=0.
   for i in range(64):
    draws=[];r=[];c=[]
    for draw in range(4):
     inp=[jnp.asarray(x) for x in [history[i:i+1],act[i:i+1,:64],act[i:i+1,64:]]];key=jax.random.fold_in(jax.random.PRNGKey(SEED+99),i*4+draw);out,after=eval_fn(v,key,*inp);jax.block_until_ready(out);finite((out,after));assert set(after)==set(v),'Evaluation created parameters'
     if i==0 and draw==0:
      again,after2=eval_fn(v,key,*inp);jax.block_until_ready(again);replay_max=max(float(np.max(abs(np.asarray(out[k])-np.asarray(again[k])))) for k in out);assert replay_max==0
      assert all(np.array_equal(np.asarray(after[k]),np.asarray(v[k])) for k in v),'Evaluation mutated weights'
     draws.append(np.asarray(out['prediction'])[0]);r.append(np.asarray(out['reconstruction'])[0]);c.append(np.asarray(out['context'])[0])
    prediction.append(draws);reconstruction.append(np.mean(r,axis=0));contexts.append(np.mean(c,axis=0))
   draws=np.asarray(prediction);mean=draws.mean(1);reconstruction=np.asarray(reconstruction);errors=(mean.astype(np.float64)-truth.astype(np.float64))**2;draw_errors=(draws.astype(np.float64)-truth[:,None].astype(np.float64))**2;persistence=(history[:,-1:,None,:].reshape(64,1,16).astype(np.float64)-truth.astype(np.float64))**2
   rows=[]
   for f in [0,1]:
    ids=family==f;mse=float(errors[ids].mean());pers=float(persistence[ids].mean());ref=next(x for x in references if x['family']==f)['mse'];rows.append(dict(family=f,mse=mse,per_draw_mse=float(draw_errors[ids].mean()),persistence_mse=pers,persistence_ratio=mse/pers,lagged_reference_mse=ref,lagged_ratio=mse/ref,reconstruction_mse=float(((reconstruction[ids].astype(np.float64)-history[ids].astype(np.float64))**2).mean()),competence_gate=mse<.9*pers and mse<=1.25*ref))
   ck=ART/f'step{step:04d}.npz'
   with ck.open('xb') as f:np.savez_compressed(f,**{k:np.asarray(x) for k,x in model_vars(variables).items()})
   state_receipt=save_full_state(step)
   pp=payload/f'validation_step{step:04d}.npz'
   with pp.open('xb') as f:np.savez_compressed(f,predictions=draws,mean_prediction=mean,truth=truth,history=history,reconstruction=reconstruction,context=np.asarray(contexts),family=family)
   row=dict(step=step,mse=float(errors.mean()),families=rows,checkpoint_path=str(ck),checkpoint_sha256=sha(ck),full_state_path=state_receipt['full_state_path'],full_state_sha256=state_receipt['full_state_sha256'],payload_path=str(pp),payload_sha256=sha(pp),validation_seconds=time.perf_counter()-eval_start,replay_max=replay_max,optimizer_updates=step,numpy_rng_state=rng.bit_generator.state,next_update_prng_seed=SEED+step+1,elapsed_seconds=time.perf_counter()-started)
   dump(OUT/f'VALIDATION_STEP{step:04d}_V1.json',row);all_rows.append(row);event('factual_validation_complete',step=step,mse=row['mse'],families=rows,seconds=row['validation_seconds'])
   if row['mse']<best:best=row['mse'];chosen=row
  if step==TOTAL:break
  i=int(rng.integers(len(train['actions'])));s=int(rng.integers(0,129));d=make_data(train['observations'][i:i+1,s:s+64],train['actions'][i:i+1,s:s+64]);st=time.perf_counter();out,variables=train_fn(variables,jax.random.PRNGKey(SEED+step+1),d);jax.block_until_ready(out);state,outs,metrics=out
  finite((state,outs,variables));allowed={'reward_pos_loss','reward_pos_acc','cont_neg_loss','cont_neg_acc'};scalar={}
  for k,x in metrics.items():
   a=np.asarray(x)
   if not np.isfinite(a).all():assert k in allowed and a.shape==() and np.isnan(a).all(),k
   if a.shape==():scalar[k]=float(a) if np.isfinite(a) else None
  if step+1 in oldprogress:
   differences={k:abs(float(v)-float(oldprogress[step+1][k])) for k,v in scalar.items() if v is not None};delta=max(differences.values());replay_comparisons.append(dict(step=step+1,max_scalar_difference=delta));event('original_prefix_replay_compared',step=step+1,max_scalar_difference=delta)
   assert delta<=1e-6,('Recovered prefix differs from original source run',step+1,delta)
  if step+1==1 or ((step+1)%250==0 and step+1 not in CHECKS):save_full_state(step+1)
  if step<2 or (step+1)%50==0:
   event('factual_train_progress',step=step+1,trajectory=i,window_start=s,step_seconds=time.perf_counter()-st,interval_seconds=time.perf_counter()-time_since,metrics=scalar);time_since=time.perf_counter()
 qualify_source();dump(OUT/'RESULTS_V1.json',dict(status='NATIVE_FACTUAL_TRANSFER_RECOVERY_COMPLETE',seed=SEED,optimizer_updates=TOTAL,original_incomplete_optimizer_calls=500,total_actual_optimizer_calls=TOTAL+500,replayed_prefix_calls=500,replay_comparisons=replay_comparisons,scientific_fit_trajectories=1,completed_runs=1,original_incomplete_runs=1,selected=chosen,validation_trace=all_rows,competence_gate=all(x['competence_gate'] for x in chosen['families']),alignment_same_rng_explicit_step_check='UNQUALIFIED_NOT_EXECUTED_IN_TRAINING',full_mechanism_evaluation_qualification=False,source_files_unchanged=True,protocol_sha256=sha(ROOT/'FACTUAL_FIT_PROTOCOL_V1.md'),source_revision=source['source_revision'],checkpoint_selection='developmental VAL aggregate mean-forecast MSE',intervention_queries_evaluated=False,independent_confirmation=False,paper_reproduction=False,global_mechanism_fidelity_proven=False,elapsed_seconds=time.perf_counter()-started));event('factual_fit_complete',updates=TOTAL,selected_step=chosen['step'],competence_gate=all(x['competence_gate'] for x in chosen['families']))
except Exception as e:
 dump(OUT/'FAILURE_V1.json',dict(exception_type=type(e).__name__,exception=str(e),traceback=traceback.format_exc(),status='NATIVE_FACTUAL_TRANSFER_RECOVERY_INCOMPLETE'));raise
