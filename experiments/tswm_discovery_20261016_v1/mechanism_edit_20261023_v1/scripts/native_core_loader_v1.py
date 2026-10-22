"""Load unchanged native core definitions without unrelated RL/environment imports."""
import pathlib,sys,types,ast,json,re,io,logging,hashlib,importlib
import numpy as np,jax,jax.numpy as jnp,ruamel.yaml
ROOT=pathlib.Path(__file__).resolve().parents[1]
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def qualify_source():
 q=read(ROOT/'SOURCE_ACQUISITION_V1.json');src=pathlib.Path(q['source_root'])
 for row in read(ROOT/'SOURCE_AST_AUDIT_V1.json')['files']:assert sha(src/row['path'])==row['sha256'],row['path']
 return src,q
def load_core():
 src,q=qualify_source();pkg=types.ModuleType('_dali_native_core');pkg.__path__=[str(src/'dreamerv3_compat/dreamerv3')];sys.modules[pkg.__name__]=pkg
 nets=importlib.import_module(pkg.__name__+'.nets');ju=importlib.import_module(pkg.__name__+'.jaxutils');nj=importlib.import_module(pkg.__name__+'.ninjax')
 p=src/'dreamerv3_compat/dreamerv3/embodied/core/config.py';node=next(x for x in ast.parse(p.read_text()).body if isinstance(x,ast.ClassDef) and x.name=='Config');ns=dict(re=re,json=json,io=io);exec(compile(ast.Module(body=[node],type_ignores=[]),str(p),'exec'),ns)
 cfg=ruamel.yaml.YAML(typ='safe').load((src/'contextual_mbrl/dreamer/configs.yaml').read_text());config=ns['Config'](cfg['defaults']).update(cfg['carl']).update(cfg['enc_obs_dec_obs_ctxencoder_transformer_grssm']).update({'ctx_encoder.crossmodal':True,'batch_size':1,'batch_length':64,'task':'carl_dmc_ball_in_cup','jax.platform':'cpu'})
 p=src/'dreamerv3_compat/dreamerv3/agent.py';nodes=[x for x in ast.parse(p.read_text()).body if (isinstance(x,ast.ClassDef) and x.name=='WorldModel') or (isinstance(x,ast.FunctionDef) and x.name=='nonzero_in')];assert len(nodes)==2
 ns=dict(jax=jax,jnp=jnp,np=np,nj=nj,nets=nets,jaxutils=ju,cast=ju.cast_to_compute,sg=lambda x:jax.tree_util.tree_map(jax.lax.stop_gradient,x),tree_map=jax.tree_util.tree_map,logger=logging.getLogger('native-core-transfer'));exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),ns)
 wm=ns['WorldModel']({'obs':types.SimpleNamespace(shape=(16,))},{'action':types.SimpleNamespace(shape=(2,))},config,name='wm')
 return wm,nj,config,q
