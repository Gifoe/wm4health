"""Actual unchanged native loss gradient routing; no optimizer calls or model claim."""
import os
os.environ.update(JAX_PLATFORMS='cpu',OMP_NUM_THREADS='2',XLA_FLAGS='--xla_cpu_multi_thread_eigen=false')
import sys,pathlib,json,datetime,traceback
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from native_core_loader_v1 import ROOT,read,sha,load_core,qualify_source
import numpy as np,jax,jax.numpy as jnp
OUT=ROOT/'crossmodal_gradient_contract_v1';OUT.mkdir(exist_ok=False)
def dump(n,j):
 with (OUT/n).open('x',encoding='utf-8') as f:json.dump(j,f,indent=2,allow_nan=False)
try:
 wm,nj,config,source=load_core();fn=nj.pure(lambda c,s:wm.ctx_encoder.reconstruct_wm_state_loss(c,s));key=jax.random.PRNGKey(202610239);rng=np.random.default_rng(202610239);ctx=jnp.asarray(rng.normal(size=(2,8)),jnp.float32);state=jnp.asarray(rng.normal(size=(2,1536)),jnp.float32)
 values,variables=fn({},key,ctx,state);objective=lambda v,c,s:fn(v,key,c,s)[0].sum();loss,(gv,gc,gs)=jax.value_and_grad(objective,argnums=(0,1,2))(variables,ctx,state);jax.block_until_ready((loss,gv,gc,gs))
 assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves((loss,variables,gv,gc,gs)))
 projection={k:float(np.max(abs(np.asarray(g)))) for k,g in gv.items() if '/recon_wm/' in k};reverse={k:float(np.max(abs(np.asarray(g)))) for k,g in gv.items() if '/recon_ctxc/' in k};assert projection and reverse
 assert all(x==0 for x in projection.values());assert max(reverse.values())>0 and float(np.max(abs(np.asarray(gc))))>0 and float(np.max(abs(np.asarray(gs))))>0
 store=read(ROOT.parent/'STORAGE_LOCATION_V1.json');art=pathlib.Path(store['artifact_root'])/ROOT.name/'crossmodal_gradient_contract_v1';art.mkdir(parents=True,exist_ok=False);path=art/'native_loss_gradients_v1.npz'
 arrays={'context':np.asarray(ctx),'wm_state':np.asarray(state),'loss':np.asarray(loss),'context_gradient':np.asarray(gc),'wm_state_gradient':np.asarray(gs)};arrays.update({'parameter/'+k:np.asarray(v) for k,v in variables.items()});arrays.update({'gradient/'+k:np.asarray(g) for k,g in gv.items()})
 with path.open('xb') as f:np.savez_compressed(f,**arrays)
 qualify_source();result=dict(status='NATIVE_CROSSMODAL_PROJECTION_GRADIENT_ROUTE_EXECUTED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_revision=source['source_revision'],source_files_unchanged=True,fixture_shapes=dict(context=[2,8],wm_state=[2,1536]),loss=float(loss),forward_projection_gradient_maxima=projection,reverse_projection_gradient_maxima=reverse,context_gradient_max=float(np.max(abs(np.asarray(gc)))),wm_state_gradient_max=float(np.max(abs(np.asarray(gs)))),optimizer_updates=0,scientific_fits=0,trained_checkpoint_used=False,forecaster_failure_explained=False,physical_edit_evaluated=False,payload_path=str(path),payload_sha256=sha(path),protocol_sha256=sha(ROOT/'PAPER_SOURCE_OBJECTIVE_BOUNDARY_V1.md'))
 dump('RESULTS_V1.json',result);print(json.dumps(result),flush=True)
except Exception as e:dump('FAILURE_V1.json',dict(type=type(e).__name__,exception=str(e),traceback=traceback.format_exc()));raise
