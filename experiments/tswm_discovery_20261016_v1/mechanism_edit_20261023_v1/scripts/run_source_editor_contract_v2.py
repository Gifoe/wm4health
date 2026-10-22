"""Execute original recorder closures with JAX and routing spies; no learned model."""
import os
os.environ['JAX_PLATFORMS']='cpu';os.environ['OMP_NUM_THREADS']='2';os.environ['WANDB_MODE']='disabled'
import pathlib,json,hashlib,ast,types,importlib.util,sys,datetime
import numpy as np,jax,jax.numpy as jnp
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'editor_contract_recovery_v1';OUT.mkdir(exist_ok=False)
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def dump(n,j):
 with (OUT/n).open('x',encoding='utf-8') as f:json.dump(j,f,indent=2,allow_nan=False)
q=read(ROOT/'SOURCE_ACQUISITION_V1.json');SRC=pathlib.Path(q['source_root']);manifest=read(ROOT/'SOURCE_AST_AUDIT_V1.json')
for x in manifest['files']:assert sha(SRC/x['path'])==x['sha256']
storage=read(ROOT.parent/'STORAGE_LOCATION_V1.json');ART=pathlib.Path(storage['artifact_root'])/ROOT.name/'source_editor_contract_recovery_v1';ART.mkdir(parents=True,exist_ok=False)
spec=importlib.util.spec_from_file_location('dali_native_ninjax',SRC/'dreamerv3_compat/dreamerv3/ninjax.py');nj=importlib.util.module_from_spec(spec);sys.modules[spec.name]=nj;spec.loader.exec_module(nj)
T=64;D=8;DIM=6;SEED=20261023
dump('MATRIX_V1.json',dict(source_revision=q['source_revision'],fixtures=['native_broadcast','axis_only_non_native'],batches_by_variant={'default':[1,2],'obs_dataset':[10]},input_lengths_by_variant={'default':64,'obs_dataset':65},time=T,context_dimensions=D,selected_dimension=DIM,seed=SEED,protocol_sha256=sha(ROOT/'SOURCE_EDITOR_CONTRACT_PROTOCOL_V1.md'),axis_fixture_addendum_sha256=sha(ROOT/'AXIS_FIXTURE_ADDENDUM_V1.md'),recovery_contract_sha256=sha(ROOT/'RECORDER_HARNESS_RECOVERY_V1.md'),learned_models_executed=False,optimizer_updates=0,source_extraction='unmodified original _wrap_dream_agent AST; native ninjax pure/RNG; video identity spy'))
class Dist:
 def __init__(self,x):self.x=x
 def mode(self):return self.x
 def sample(self,seed=None):return self.x
class SpyRSSM:
 _action_clip=1.;_classes=0;_stoch=2;_add_context_prior=False;_kw={}
 def __init__(self,owner):self.owner=owner
 def observe(self,embed,actions,is_first,dcontext=None):
  b,t=actions.shape[:2];self.owner.observations.append(dict(length=t,context=np.asarray(dcontext).copy()));s=dict(deter=jnp.broadcast_to(jnp.arange(t,dtype=jnp.float32)[None,:,None],(b,t,2)),stoch=jnp.zeros((b,t,2)))
  return s,s
 def imagine(self,actions,state,dcontext=None):
  self.owner.imaginings.append(dict(context=np.asarray(dcontext).copy(),actions=np.asarray(actions).copy(),root=np.asarray(state['deter']).copy()));b,t=actions.shape[:2];return dict(deter=jnp.zeros((b,t,2)),stoch=jnp.zeros((b,t,2)))
 def get(self,name,constructor,**kw):
  def op(x):
   if name=='img_in':self.owner.manual_contexts.append(np.asarray(x[...,-D:]).copy())
   return x
  return op
 def _gru(self,x,prev):return x,jnp.zeros_like(prev)
 def _stats(self,name,x):return dict(mean=jnp.zeros((*x.shape[:-1],2)),std=jnp.ones((*x.shape[:-1],2)))
 def get_dist(self,stats):return Dist(stats['mean'])
class SpyWM:
 def __init__(self,context):
  self.context=context;self.imaginings=[];self.observations=[];self.manual_contexts=[];self.encoder_inputs=[];self.rssm=SpyRSSM(self);self.heads={'decoder':self.decode,'cont':self.cont}
 def encoder(self,data):return jnp.zeros((*data['obs'].shape[:2],2))
 def ctx_encoder(self,data):
  b,t=data['obs'].shape[:2];self.encoder_inputs.append(dict(observations=t,actions=data['action'].shape[1]));return self.context[:b,:t]
 def initial(self,b):return None
 def decode(self,state):
  b,t=state['deter'].shape[:2];return dict(image=Dist(jnp.zeros((b,t,2,2,3))),obs=Dist(jnp.zeros((b,t,16))))
 def cont(self,state):return Dist(jnp.ones(state['deter'].shape[:2]))
class SpyPolicy:
 def policy(self,features,state,dcontext=None):return dict(action=Dist(jnp.zeros((*dcontext.shape[:2],2)))),state
class SpyAgent:
 def __init__(self,context):
  self.wm=SpyWM(context);self.task_behavior=SpyPolicy();self.config=types.SimpleNamespace(batch_length=T,task='carl_dmc_ball_in_cup',ctx_encoder=types.SimpleNamespace(inputs=['obs','action'],linear_ctx_out=types.SimpleNamespace(hunits=D)))
 def preprocess(self,data):return dict(data)
 def policy_initial(self,b):return None,None
def descriptor(delta):
 a=np.asarray(delta,dtype=np.float64);changed=abs(a)>0;return dict(max_abs=float(abs(a).max()),changed_batches=np.flatnonzero(changed.any(axis=(1,2))).tolist(),changed_times=np.flatnonzero(changed.any(axis=(0,2))).tolist(),changed_dimensions=np.flatnonzero(changed.any(axis=(0,1))).tolist(),time_variation=float(abs(a-a[:,:1]).max()),nonzero_elements=int(changed.sum()))
payload={};records=[];source_contracts=[]
paths=sorted((SRC/'contextual_mbrl/dreamer').glob('*counterfactual*.py'))
for p in paths:
 text=p.read_text(encoding='utf-8');tree=ast.parse(text);functions=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='_wrap_dream_agent'];assert len(functions)==1
 node=functions[0];namespace=dict(tree_map=jax.tree_util.tree_map,jax=jax,jnp=jnp,np=np,nj=nj,sg=jax.lax.stop_gradient,Linear=object,jaxutils=types.SimpleNamespace(video_grid=lambda x:x))
 exec(compile(ast.Module(body=[node],type_ignores=[]),str(p),'exec'),namespace);wrapper=namespace['_wrap_dream_agent'];source_contracts.append(dict(path=p.relative_to(SRC).as_posix(),sha256=sha(p),lines=[node.lineno,node.end_lineno],wrapper_source_sha256=hashlib.sha256(ast.get_source_segment(text,node).encode()).hexdigest()))
 for kind in ['native_broadcast','axis_only_non_native']:
  for B in ([10] if p.name=='record_counterfactual_plausibility_obs_dataset.py' else [1,2]):
   input_T=65 if p.name=='record_counterfactual_plausibility_obs_dataset.py' else T
   base=jnp.arange(1,D+1,dtype=jnp.float32)[None,None,:]/8+jnp.arange(B,dtype=jnp.float32)[:,None,None]/2;context=jnp.broadcast_to(base,(B,input_T,D))
   if kind=='axis_only_non_native':context=context+jnp.arange(input_T,dtype=jnp.float32)[None,:,None]/8
   agent=SpyAgent(context);fn=wrapper(agent,DIM) if len(node.args.args)>1 else wrapper(agent)
   data=dict(obs=jnp.broadcast_to(jnp.arange(input_T,dtype=jnp.float32)[None,:,None],(B,input_T,16)),action=jnp.zeros((B,input_T,2)),is_first=jnp.zeros((B,input_T),bool).at[:,0].set(True),image=jnp.zeros((B,input_T,2,2,3)))
   result,state=nj.pure(fn)({},jax.random.PRNGKey(SEED),data);assert not state
   stream=agent.wm.imaginings
   if agent.wm.manual_contexts:
    assert len(agent.wm.manual_contexts)==8*50
    context_sequences=np.concatenate(agent.wm.manual_contexts,axis=1).reshape(B,8,50,D).transpose(1,0,2,3)
   else:context_sequences=np.asarray([x['context'] for x in stream])
   key=p.stem+'_'+kind+'_B'+str(B);payload[key+'_contexts']=context_sequences;payload[key+'_base_context']=np.asarray(context);descs=[]
   for i,a in enumerate(context_sequences):descs.append(dict(call=i,**descriptor(a-np.asarray(context[:,:a.shape[1]]))))
   records.append(dict(file=p.name,fixture=kind,batch=B,input_length=input_T,imagination_calls=len(stream),manual_prior_steps=len(agent.wm.manual_contexts),encoder_calls=agent.wm.encoder_inputs,observe_lengths=[x['length'] for x in agent.wm.observations],roots=[np.unique(x['root']).tolist() for x in stream],context_shapes=list(context_sequences.shape),temporal_sigma=np.asarray(context[0].std(0)).tolist(),pooled_sigma=np.asarray(context.reshape(-1,D).std(0)).tolist(),edits=descs,output_keys=len(result),native_model_executed=False))
   print(json.dumps(dict(file=p.name,fixture=kind,batch=B,calls=len(descs),nonzero_edit_calls=sum(x['nonzero_elements']>0 for x in descs),max_abs=max(x['max_abs'] for x in descs))),flush=True)
# Ordinary explicit-axis, fixed-scale, fixed-sign tensor controls; no method claim.
controls=[]
for B in [1,2]:
 ctx=jnp.broadcast_to(jnp.arange(1,D+1,dtype=jnp.float32)[None,None,:]/8+jnp.arange(B,dtype=jnp.float32)[:,None,None]/2,(B,T,D));edited=ctx.at[:,:,DIM].add(.5);d=descriptor(np.asarray(edited-ctx));assert d['changed_dimensions']==[DIM] and len(d['changed_times'])==T and d['time_variation']==0 and d['max_abs']==.5;controls.append(dict(batch=B,**d));payload['control_B'+str(B)]=np.asarray(edited-ctx)
for x in manifest['files']:assert sha(SRC/x['path'])==x['sha256']
path=ART/'recorder_context_routing_v1.npz'
with path.open('xb') as f:np.savez_compressed(f,**payload)
dump('RESULTS_V1.json',dict(status='NATIVE_RECORDER_CLOSURES_EXECUTED_WITH_CONTEXT_ROUTING_SPIES',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_revision=q['source_revision'],source_contracts=source_contracts,records=records,ordinary_controls=controls,payload=dict(path=str(path),sha256=sha(path)),protocol_sha256=sha(ROOT/'SOURCE_EDITOR_CONTRACT_PROTOCOL_V1.md'),axis_fixture_addendum_sha256=sha(ROOT/'AXIS_FIXTURE_ADDENDUM_V1.md'),recovery_contract_sha256=sha(ROOT/'RECORDER_HARNESS_RECOVERY_V1.md'),ninjax_sha256=sha(SRC/'dreamerv3_compat/dreamerv3/ninjax.py'),learned_model_executed=False,optimizer_updates=0,paper_results_invalidated=False,accepted_final_directions=0))
print(json.dumps(dict(status='NATIVE_RECORDER_CONTRACT_PREFLIGHT_COMPLETE',variants=len(paths),cases=len(records),new_training=0,learned_model_executed=False)),flush=True)
