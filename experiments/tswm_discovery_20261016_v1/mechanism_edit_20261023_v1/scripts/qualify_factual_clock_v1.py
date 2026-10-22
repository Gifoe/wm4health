"""Match native scanned dynamics to explicit native steps at known action clocks."""
import os
os.environ.update(JAX_PLATFORMS='cpu',OMP_NUM_THREADS='2',XLA_FLAGS='--xla_cpu_multi_thread_eigen=false')
import sys,pathlib,argparse,json,datetime,traceback
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from native_core_loader_v1 import ROOT,read,sha,load_core,qualify_source
import numpy as np,jax,jax.numpy as jnp
p=argparse.ArgumentParser();p.add_argument('--step',type=int,choices=[0,500,1000,2000],required=True);args=p.parse_args();OUT=ROOT/f'factual_clock_step{args.step:04d}_v1';OUT.mkdir(exist_ok=False)
def dump(n,x):
 with (OUT/n).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
try:
 row=read(ROOT/'factual_fit_v1'/f'VALIDATION_STEP{args.step:04d}_V1.json');assert sha(row['checkpoint_path'])==row['checkpoint_sha256'];wm,nj,config,source=load_core()
 with np.load(row['checkpoint_path'],allow_pickle=False) as z:params={k:jnp.asarray(z[k]) for k in z.files if 'model_opt' not in k}
 matrix=read(ROOT/'factual_fit_v1/MATRIX_V1.json');val=matrix['data']['val'];assert sha(val['path'])==val['sha256']
 with np.load(val['path'],allow_pickle=False) as z:h=jnp.asarray(z['observations'][:1,48:48+65]);a=jnp.asarray(z['actions'][:1,48:48+96])
 def encoded(h,a):
  e=wm.encoder({'obs':h});c=wm.ctx_encoder({'obs':h[:,:64],'action':a[:,:64],'embed':e[:,:64] if 'embed' in config.ctx_encoder.inputs else None});return e,c,wm.rssm.initial(1)
 def run(fn,*x,key):
  out,after=fn(params,key,*x,create=False,modify=False);jax.block_until_ready(out);assert set(after)==set(params)
  for leaf in jax.tree_util.tree_leaves(out):assert np.isfinite(np.asarray(leaf)).all()
  return out
 key=jax.random.PRNGKey(202610237);encoder=jax.jit(nj.pure(encoded));embed,ctx,initial=run(encoder,h,a,key=key);ctx65=jnp.broadcast_to(ctx[:,:1],(1,65,8));previous=jnp.concatenate([jnp.zeros_like(a[:,:1]),a[:,:64]],1);first=jnp.zeros((1,65),bool).at[:,0].set(True)
 obs_scan=jax.jit(nj.pure(lambda e,a,f,s,c:wm.rssm.observe(e,a,f,s,dcontext=c)));obs_step=jax.jit(nj.pure(lambda s,a,e,f,c:wm.rssm.obs_step(s,a,e,f,dcontext=c)))
 post,prior=run(obs_scan,embed,previous,first,initial,ctx65,key=key);stepkeys=jax.random.split(key,66)[1:];state=initial;manual=[]
 for i in range(65):state,_=run(obs_step,state,previous[:,i],embed[:,i],first[:,i],ctx65[:,i],key=stepkeys[i]);manual.append(state)
 manual_post={k:jnp.stack([x[k] for x in manual],1) for k in post};post_error={k:float(jnp.max(abs(post[k]-manual_post[k]))) for k in post};assert max(post_error.values())<=1e-5 and post_error['stoch']==0,post_error
 root={k:x[:,-1] for k,x in post.items()};ctx32=jnp.broadcast_to(ctx[:,:1],(1,32,8));future=a[:,64:]
 imagine=jax.jit(nj.pure(lambda a,s,c:wm.rssm.imagine(a,s,dcontext=c)));img_step=jax.jit(nj.pure(lambda s,a,c:wm.rssm.img_step(s,a,dcontext=c)));decode=jax.jit(nj.pure(lambda states,c:wm.heads['decoder']({**states,'context':c})['obs'].mode()))
 imagined=run(imagine,future,root,ctx32,key=key);state=root;manual=[];stepkeys=jax.random.split(key,33)[1:]
 for i in range(32):state=run(img_step,state,future[:,i],ctx32[:,i],key=stepkeys[i]);manual.append(state)
 manual_prior={k:jnp.stack([x[k] for x in manual],1) for k in imagined};prior_error={k:float(jnp.max(abs(imagined[k]-manual_prior[k]))) for k in imagined};assert max(prior_error.values())<=1e-5 and prior_error['stoch']==0,prior_error
 pred=run(decode,imagined,ctx32,key=key);manual_pred=run(decode,manual_prior,ctx32,key=key);decoded_error=float(jnp.max(abs(pred-manual_pred)));assert decoded_error<=1e-5
 altered=future.at[:,24:,0].set(.8);other=run(imagine,altered,root,ctx32,key=key);other_pred=run(decode,other,ctx32,key=key);early=max(float(jnp.max(abs(imagined[k][:,:24]-other[k][:,:24]))) for k in imagined);early=max(early,float(jnp.max(abs(pred[:,:24]-other_pred[:,:24]))));assert early<=1e-6,early
 store=read(ROOT.parent/'STORAGE_LOCATION_V1.json');ART=pathlib.Path(store['artifact_root'])/ROOT.name/OUT.name;ART.mkdir(parents=True,exist_ok=False);path=ART/'clock_replay_v1.npz';arrays={}
 for prefix,states in [('scanned_post',post),('explicit_post',manual_post),('scanned_prior',imagined),('explicit_prior',manual_prior),('changed_plan_prior',other)]:arrays.update({prefix+'/'+k:np.asarray(v) for k,v in states.items()})
 arrays.update(prediction=np.asarray(pred),explicit_prediction=np.asarray(manual_pred),changed_plan_prediction=np.asarray(other_pred))
 with path.open('xb') as f:np.savez_compressed(f,**arrays)
 qualify_source();result=dict(status='NATIVE_FILTER_AND_IMAGINATION_CLOCK_QUALIFIED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),checkpoint_step=args.step,checkpoint_sha256=row['checkpoint_sha256'],posterior_errors=post_error,prior_errors=prior_error,decoded_error=decoded_error,future_change_early_max=early,action_clock=dict(observed_states=[0,64],executed_actions=[0,63],future_actions=[64,95],future_targets=[65,96]),source_files_unchanged=True,optimizer_updates=0,payload_path=str(path),payload_sha256=sha(path),protocol_sha256=sha(ROOT/'FACTUAL_CLOCK_QUALIFICATION_V1.md'),physical_prediction_accuracy_established=False)
 dump('RESULTS_V1.json',result);print(json.dumps(result),flush=True)
except Exception as e:dump('FAILURE_V1.json',dict(exception_type=type(e).__name__,exception=str(e),traceback=traceback.format_exc()));raise
