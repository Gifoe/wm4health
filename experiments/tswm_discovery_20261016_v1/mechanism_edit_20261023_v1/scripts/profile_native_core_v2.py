"""Exact native model core CPU timing/gradient preflight; no scientific fit."""
import os
os.environ['JAX_PLATFORMS']='cpu';os.environ['OMP_NUM_THREADS']='4';os.environ['TF_NUM_INTRAOP_THREADS']='4';os.environ['TF_NUM_INTEROP_THREADS']='1';os.environ['XLA_FLAGS']='--xla_cpu_multi_thread_eigen=false';os.environ['WANDB_MODE']='disabled'
import pathlib,sys,types,ast,json,re,io,logging,hashlib,time,datetime,importlib,traceback
import numpy as np,jax,jax.numpy as jnp,ruamel.yaml
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'native_core_profile_recovery_v1';OUT.mkdir(exist_ok=False)
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def dump(n,j):
 with (OUT/n).open('x',encoding='utf-8') as f:json.dump(j,f,indent=2,allow_nan=False)
def event(stage,**kw):
 row=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),stage=stage,**kw)
 with (OUT/'trace.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
 print(json.dumps(row),flush=True)
q=read(ROOT/'SOURCE_ACQUISITION_V1.json');SRC=pathlib.Path(q['source_root']);manifest=read(ROOT/'SOURCE_AST_AUDIT_V1.json')
dump('MATRIX_V1.json',dict(protocol_sha256=sha(ROOT/'NATIVE_CORE_PROFILE_PROTOCOL_V1.md'),metric_domain_recovery_sha256=sha(ROOT/'CORE_PROFILE_METRIC_DOMAIN_RECOVERY_V1.md'),source_revision=q['source_revision'],batch=1,sequence=64,compute_dtype='float32 CPU',qualified_updates=2,discard_updates=True,full_paper_reproduction=False))
try:
 for row in manifest['files']:assert sha(SRC/row['path'])==row['sha256']
 pkg=types.ModuleType('_dali_native_core');pkg.__path__=[str(SRC/'dreamerv3_compat/dreamerv3')];sys.modules[pkg.__name__]=pkg
 nets=importlib.import_module(pkg.__name__+'.nets');ju=importlib.import_module(pkg.__name__+'.jaxutils');nj=importlib.import_module(pkg.__name__+'.ninjax');event('native_core_modules_imported',devices=[str(x) for x in jax.devices()])
 p=SRC/'dreamerv3_compat/dreamerv3/embodied/core/config.py';tree=ast.parse(p.read_text());node=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='Config');ns=dict(re=re,json=json,io=io);exec(compile(ast.Module(body=[node],type_ignores=[]),str(p),'exec'),ns);Config=ns['Config']
 allcfg=ruamel.yaml.YAML(typ='safe').load((SRC/'contextual_mbrl/dreamer/configs.yaml').read_text());config=Config(allcfg['defaults']).update(allcfg['carl']).update(allcfg['enc_obs_dec_obs_ctxencoder_transformer_grssm']).update({'ctx_encoder.crossmodal':True,'batch_size':1,'batch_length':64,'task':'carl_dmc_ball_in_cup','jax.platform':'cpu'})
 dump('CONFIG_V1.json',dict(config));event('config_resolved',rssm=dict(config.rssm),model_opt=dict(config.model_opt),context_network=config.ctx_encoder.network_type)
 p=SRC/'dreamerv3_compat/dreamerv3/agent.py';tree=ast.parse(p.read_text());nodes=[x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='WorldModel' or isinstance(x,ast.FunctionDef) and x.name=='nonzero_in'];assert len(nodes)==2
 ns=dict(jax=jax,jnp=jnp,np=np,nj=nj,nets=nets,jaxutils=ju,cast=ju.cast_to_compute,sg=lambda x:jax.tree_util.tree_map(jax.lax.stop_gradient,x),tree_map=jax.tree_util.tree_map,logger=logging.getLogger('native-core-profile'));exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),ns)
 observations={'obs':types.SimpleNamespace(shape=(16,))};actions={'action':types.SimpleNamespace(shape=(2,))};wm=ns['WorldModel'](observations,actions,config,name='wm')
 datarec=next(x for x in read(ROOT.parent/'cycle88/outputs/PLAN_DATA88_V1.json')['files'] if x['split']=='train');assert sha(datarec['path'])==datarec['sha256']
 with np.load(datarec['path'],allow_pickle=False) as z:x=z['observations'][:1,:64].copy();a=z['actions'][:1,:64].copy()
 data=dict(obs=jnp.asarray(x),action=jnp.asarray(a),reward=jnp.zeros((1,64)),cont=jnp.ones((1,64)),is_first=jnp.zeros((1,64),bool).at[:,0].set(True),is_last=jnp.zeros((1,64),bool),is_terminal=jnp.zeros((1,64),bool))
 dump('INPUT_FIXTURE_V1.json',dict(training_data_sha256=datarec['sha256'],rows=[0],observation_times=[0,64],validation_or_eval_read=False,reward_and_continuation_are_profiling_placeholders=True))
 function=jax.jit(nj.pure(lambda data:wm.train(data,wm.initial(1))));variables={};times=[];event('compile_and_first_update_start')
 previous_parameters=None;unavailable=[];parameter_update_max=None
 for k in range(2):
  start=time.perf_counter();out,variables=function(variables,jax.random.PRNGKey(20261023+k),data);jax.block_until_ready((out,variables));elapsed=time.perf_counter()-start
  state,outs,metrics=out
  for leaf in jax.tree_util.tree_leaves((state,outs,variables)):
   if hasattr(leaf,'dtype'):assert np.isfinite(np.asarray(leaf)).all()
  allowed={'reward_pos_loss','reward_pos_acc','cont_neg_loss','cont_neg_acc'}
  assert np.all(np.asarray(data['reward'])<=.1) and np.all(np.asarray(data['cont'])>.5)
  scalar={};missing=[]
  for name,value in metrics.items():
   array=np.asarray(value)
   if not np.isfinite(array).all():
    assert name in allowed and array.shape==() and np.isnan(array).all(),(name,array.shape)
    missing.append(name)
   if array.shape==():scalar[str(name)]=None if name in missing else float(array)
  current_parameters={name:np.asarray(value).copy() for name,value in variables.items() if 'model_opt' not in name and hasattr(value,'shape')}
  if previous_parameters is not None:
   parameter_update_max=max(float(np.max(abs(value-previous_parameters[name]))) for name,value in current_parameters.items());assert parameter_update_max>0
  previous_parameters=current_parameters;unavailable.append(missing);times.append(elapsed);event('qualification_update_complete',update=k+1,seconds=elapsed,scalar_metrics=scalar,unavailable_empty_class_metrics=missing,parameter_update_max=parameter_update_max)

 param_count=sum(int(np.prod(v.shape)) for n,v in variables.items() if 'model_opt' not in n and hasattr(v,'shape'))
 registration={n:dict(shape=list(v.shape),dtype=str(v.dtype)) if hasattr(v,'shape') else dict(type=type(v).__name__) for n,v in variables.items()};dump('PARAMETER_REGISTRATION_V1.json',registration)
 for row in manifest['files']:assert sha(SRC/row['path'])==row['sha256']
 dump('RESULTS_V1.json',dict(status='NATIVE_WORLD_MODEL_CPU_CORE_EXECUTED_TWO_DISCARDED_UPDATES',source_revision=q['source_revision'],seconds=times,unavailable_empty_class_metrics=unavailable,second_update_parameter_max_abs_change=parameter_update_max,initial_failed_profile_qualification_updates=1,non_optimizer_array_parameter_elements=param_count,parameter_count_definition='all flat registered arrays excluding paths containing model_opt',qualified_optimizer_updates=2,scientific_fit_count=0,trained_model_saved=False,physical_fidelity_evaluated=False,full_rl_or_paper_reproduction=False,source_files_unchanged=True,protocol_sha256=sha(ROOT/'NATIVE_CORE_PROFILE_PROTOCOL_V1.md'),metric_domain_recovery_sha256=sha(ROOT/'CORE_PROFILE_METRIC_DOMAIN_RECOVERY_V1.md')));event('core_profile_complete',seconds=times,parameter_elements=param_count)
except Exception as e:
 dump('FAILURE_V1.json',dict(status='NATIVE_CORE_PREFLIGHT_INCOMPLETE',exception_type=type(e).__name__,exception=str(e),traceback=traceback.format_exc(),physical_fidelity_evaluated=False,scientific_fit_count=0));raise
