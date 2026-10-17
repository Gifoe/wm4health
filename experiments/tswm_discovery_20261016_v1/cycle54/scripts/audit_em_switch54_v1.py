"""Existing native EM flag control with explicit fixed-posterior derivative check."""
from common54 import *
import ctypes,pickle,time
from dataclasses import replace
from rp_ssm.datasets import TrainData
fn=ctypes.windll.kernel32.SetProcessAffinityMask;fn.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
assert fn(ctypes.c_void_p(-1),ctypes.c_size_t(0xff000000))
def function(seed,B,em):
 energy,cfg=rp_model('linear_small',seed);cfg=replace(cfg,batch_size=B,em=em)
 energy.init(jr.PRNGKey(54),TrainData(obs=(jnp.zeros((B,100,5)),)),cfg)
 return energy,jax.jit(jax.value_and_grad(lambda A,rec,x:energy.loss(({'A':A},rec),TrainData(obs=(x,)),1.,em,jr.PRNGKey(54))[0],argnums=0))
def main():
 source_check();out=OUT/'em_switch_audit_v1';assert not out.exists();out.mkdir();records=[];start=time.perf_counter()
 with np.load(data_path('linear_small')) as ar:x=ar['train_x']
 for seed in [0,1]:
  run=f'rp_linear_small_seed{seed}';r=json.loads((OUT/'runs'/run/'result.json').read_text());path=pathlib.Path(r['checkpoint']);before=sha(path);assert before==r['checkpoint_sha256'];params=pickle.loads(path.read_bytes());A=jnp.asarray(params[0]['A']);rec=params[1];xx=jnp.asarray(x)
  with np.load(ART/'reference_control_audit_v2'/f'{run}.npz') as ar:ids=ar['balanced_indices']
  modes=[];raw={}
  for em in [False,True]:
   energy,full_fn=function(seed,200,em);value,g=full_fn(A,rec,xx);_,batch_fn=function(seed,32,em);gs=[];vs=[]
   for selected in ids:
    v,gb=batch_fn(A,rec,jnp.asarray(x[selected]));gs.append(np.asarray(gb));vs.append(float(v))
   gs=np.stack(gs);g=np.asarray(g);mode={'em':em,'full_native_value':float(value),'full_A_gradient':g.tolist(),'full_alpha_derivative':float(np.sum(g*np.asarray(A))),'balanced_batch_A_gradient_mean':gs.mean(0).tolist(),'balanced_batch_alpha_derivative':float(np.sum(gs.mean(0)*np.asarray(A))),'balanced_batch_vs_full_gradient_max_abs_error':float(np.max(abs(gs.mean(0)-g)))}
   raw[str(em)+'_batch_gradients']=gs;raw[str(em)+'_batch_values']=np.asarray(vs)
   if em:
    prior=energy.model.prior.update({'A':A});chain,sites,posterior=energy.get_posterior(None,prior,[rec],TrainData(obs=(xx,)))
    def fixed_objective(alpha):
     candidate=energy.model.prior.update({'A':alpha*A});p,_=candidate.to_chain(100,None)
     kf,lg,kp=energy.get_loss_terms(p,sites,posterior)
     return -(lg-kf-kp)/(200*100)
    frozen_fn=jax.jit(jax.value_and_grad(fixed_objective));fv,fg=frozen_fn(jnp.asarray(1.));h=1e-5;fd=(float(frozen_fn(jnp.asarray(1.+h))[0])-float(frozen_fn(jnp.asarray(1.-h))[0]))/(2*h)
    mode.update(fixed_posterior_Mstep_value=float(fv),fixed_posterior_alpha_derivative=float(fg),native_vs_fixed_posterior_derivative_abs_error=abs(float(fg)-mode['full_alpha_derivative']),fixed_posterior_finite_difference_derivative=fd,fixed_posterior_FD_abs_error=abs(fd-float(fg)))
    assert mode['native_vs_fixed_posterior_derivative_abs_error']<1e-7
    assert mode['fixed_posterior_FD_abs_error']<1e-4*max(1.,abs(float(fg)))
    assert mode['balanced_batch_vs_full_gradient_max_abs_error']<1e-7
   modes.append(mode)
  forward_error=abs(modes[0]['full_native_value']-modes[1]['full_native_value']);batch_value_error=float(np.max(abs(raw['False_batch_values']-raw['True_batch_values'])));assert forward_error<1e-10 and batch_value_error<1e-10
  artifact=ART/'em_switch_audit_v1';artifact.mkdir(exist_ok=True);ap=artifact/(run+'.npz');assert not ap.exists();np.savez_compressed(ap,balanced_indices=ids,**raw)
  record={'run':run,'checkpoint_sha256':before,'protocol_sha256':sha(ROOT/'TRAIN_ONLY_EM_SWITCH54_V1.md'),'script_sha256':sha(__file__),'modes':modes,'full_forward_value_difference':forward_error,'batch_forward_value_max_difference':batch_value_error,'artifact':str(ap),'artifact_sha256':sha(ap),'checkpoint_unchanged':sha(path)==before,'latent_truth_read':False,'heldout_read':False,'optimizer_updates':0,'scope':'Native existing EM option and fixed-posterior partial-derivative control only. No assertion of trained-model quality or of reference invariance of encoder updates.'}
  assert record['checkpoint_unchanged'];records.append(record);dump(out/(run+'.json'),record);print(json.dumps(record),flush=True)
 dump(out/'summary.json',{'utc':utc(),'status':'COMPLETE_EXISTING_NATIVE_EM_SWITCH_FALSIFICATION','records':records,'runtime_seconds':time.perf_counter()-start,'source_files_unchanged':source_check(),'formal_training_unchanged':True})
if __name__=='__main__':main()
