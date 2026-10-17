"""Frozen balanced-target reference attribution; no state labels or heldout arrays."""
from common54 import *
import ctypes,pickle,time
from audit_training_objective54_v2 import objective
from per_target_auxiliary54_v1 import per_target_loss
fn=ctypes.windll.kernel32.SetProcessAffinityMask;fn.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
assert fn(ctypes.c_void_p(-1),ctypes.c_size_t(0xff000000))
def main():
 source_check();out=OUT/'reference_control_audit_v1';assert not out.exists();out.mkdir()
 with np.load(data_path('linear_small')) as ar:x=ar['train_x']
 rng=np.random.default_rng(5425);ids=np.concatenate([rng.permutation(200) for _ in range(4)]).reshape(25,32)
 assert np.array_equal(np.bincount(ids.ravel(),minlength=200),np.full(200,4));records=[];start=time.perf_counter()
 for seed in [0,1]:
  run=f'rp_linear_small_seed{seed}';r=json.loads((OUT/'runs'/run/'result.json').read_text());path=pathlib.Path(r['checkpoint']);before=sha(path);assert before==r['checkpoint_sha256'];params=pickle.loads(path.read_bytes());xx=jnp.asarray(x)
  energy,full_fn=objective('linear_small',seed,200);native_loss,native_grad=full_fn(params,xx)
  loss_fn=jax.jit(jax.value_and_grad(lambda p:per_target_loss(energy,p,xx).mean()))
  loss,g=loss_fn(params);loss_err=abs(float(loss)-float(native_loss));grad_err=float(np.max(abs(np.asarray(g[0]['A'])-np.asarray(native_grad[0]['A']))))
  assert loss_err<1e-8 and grad_err<1e-7,(loss_err,grad_err)
  full_reference=jax.jit(jax.value_and_grad(lambda p,selected:per_target_loss(energy,p,xx)[selected].mean()))
  _,batch_fn=objective('linear_small',seed,32);fixed_g=[];native_g=[];fixed_l=[];native_l=[]
  for selected in ids:
   vl,vg=full_reference(params,jnp.asarray(selected));nl,ng=batch_fn(params,jnp.asarray(x[selected]))
   fixed_g.append(np.asarray(vg[0]['A']));native_g.append(np.asarray(ng[0]['A']));fixed_l.append(float(vl));native_l.append(float(nl))
  fixed_g=np.stack(fixed_g);native_g=np.stack(native_g);g0=np.asarray(native_grad[0]['A']);A=np.asarray(params[0]['A'])
  balance_err=float(np.max(abs(fixed_g.mean(0)-g0)));balance_loss_err=abs(float(np.mean(fixed_l))-float(native_loss));assert balance_err<1e-7 and balance_loss_err<1e-8,(balance_err,balance_loss_err)
  artifact=ART/'reference_control_audit_v1';artifact.mkdir(exist_ok=True);ap=artifact/(run+'.npz');assert not ap.exists()
  np.savez_compressed(ap,balanced_indices=ids,full_reference_gradients=fixed_g,batch_reference_gradients=native_g,full_native_gradient=g0,A=A,full_reference_losses=np.asarray(fixed_l),batch_reference_losses=np.asarray(native_l))
  gm=native_g.mean(0);record={'run':run,'checkpoint_sha256':before,'artifact':str(ap),'artifact_sha256':sha(ap),'protocol_sha256':sha(ROOT/'TRAIN_ONLY_REFERENCE_CONTROL54_V1.md'),'script_sha256':sha(__file__),'decomposition_source_sha256':sha(pathlib.Path(__file__).with_name('per_target_auxiliary54_v1.py')),'native_decomposition_loss_error':loss_err,'native_decomposition_A_gradient_error':grad_err,'balanced_full_reference_gradient_max_abs_error':balance_err,'balanced_full_reference_loss_error':balance_loss_err,'full_native_alpha_derivative':float(np.sum(g0*A)),'balanced_full_reference_alpha_derivative':float(np.sum(fixed_g.mean(0)*A)),'balanced_batch_reference_alpha_derivative':float(np.sum(gm*A)),'balanced_batch_reference_gradient_cosine_vs_full':float(np.vdot(gm,g0)/(np.linalg.norm(gm)*np.linalg.norm(g0))),'balanced_target_frequency':4,'batches':25,'native_batch_size':32,'complete_reference_size':200,'heldout_read':False,'latent_truth_read':False,'optimizer_updates':0,'checkpoint_unchanged':sha(path)==before}
  assert record['checkpoint_unchanged'];records.append(record);dump(out/(run+'.json'),record);print(json.dumps(record),flush=True)
 dump(out/'summary.json',{'utc':utc(),'status':'COMPLETE_BALANCED_TRAINING_TARGET_REFERENCE_CONTROL','records':records,'runtime_seconds':time.perf_counter()-start,'source_files_unchanged':source_check(),'scope':'Reference-normalizer attribution only at two frozen training checkpoints. Complete reference auxiliary objective is not proper RPM likelihood. No training/readout/heldout changes.'})
if __name__=='__main__':main()
