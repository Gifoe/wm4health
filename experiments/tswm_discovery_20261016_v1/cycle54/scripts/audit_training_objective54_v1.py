"""Frozen completed-fit native auxiliary-gradient probe; no labels or heldout access."""
from common54 import *
import ctypes,pickle,time
from rp_ssm.datasets import TrainData
fn=ctypes.windll.kernel32.SetProcessAffinityMask;fn.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
assert fn(ctypes.c_void_p(-1),ctypes.c_size_t(0xff000000))

def objective(task,seed,B):
 energy,cfg=rp_model(task,seed);cfg.batch_size=B
 # Set the native objective's shape metadata through its own initialization path.
 energy.init(jr.PRNGKey(54),TrainData(obs=(jnp.zeros((B,100,5)),)),cfg)
 return energy,jax.jit(jax.value_and_grad(lambda params,x:energy.loss(params,TrainData(obs=(x,)),1.,False,jr.PRNGKey(54))[0]))

def main():
 source_check();out=OUT/'training_objective_audit_v1';assert not out.exists();out.mkdir()
 with np.load(data_path('linear_small')) as ar:train=np.asarray(ar['train_x'])
 assert train.shape==(200,100,5)
 indices=np.random.default_rng(54032).integers(0,200,size=(32,32));records=[];start=time.perf_counter()
 for seed in [0,1]:
  run=f'rp_linear_small_seed{seed}';r=json.loads((OUT/'runs'/run/'result.json').read_text());path=pathlib.Path(r['checkpoint']);before=sha(path);assert before==r['checkpoint_sha256']
  params=pickle.loads(path.read_bytes());energy,full_fn=objective('linear_small',seed,200)
  full_loss,full_grads=full_fn(params,jnp.asarray(train));gfull=np.asarray(full_grads[0]['A']);full_loss=float(full_loss);assert np.isfinite(full_loss) and np.isfinite(gfull).all()
  A=np.asarray(params[0]['A']);qmin=float(np.linalg.eigvalsh(np.eye(3)-((1.+1e-5)*A)@((1.+1e-5)*A).T).min());assert qmin>0
  full_scalar=jax.jit(lambda alpha:energy.loss(({'A':alpha*params[0]['A']},*params[1:]),TrainData(obs=(jnp.asarray(train),)),1.,False,jr.PRNGKey(54))[0])
  plus=float(full_scalar(jnp.asarray(1.+1e-5)));minus=float(full_scalar(jnp.asarray(1.-1e-5)));fd=(plus-minus)/2e-5;direction=float((gfull*A).sum());fderr=abs(fd-direction);assert fderr<1e-4*max(1.,abs(direction)),(fd,direction)
  _,batch_fn=objective('linear_small',seed,32);gs=[];losses=[]
  for ids in indices:
   value,grad=batch_fn(params,jnp.asarray(train[ids]));losses.append(float(value));gs.append(np.asarray(grad[0]['A']))
  gs=np.stack(gs);losses=np.array(losses);assert np.isfinite(gs).all() and np.isfinite(losses).all();gmean=gs.mean(0);gse=gs.std(0,ddof=1)/np.sqrt(len(gs));directions=np.einsum('nkl,kl->n',gs,A)
  factors=energy.model.get_factors(params[1:],(jnp.asarray(train),));P=np.asarray(factors.params['p'])[0];eta=np.asarray(factors.params['pwm'])[0];J=P+np.eye(3);V=np.linalg.inv(J);m=np.linalg.solve(J,eta[...,None])[...,0]
  mf=m.mean(0);diff=m-mf;cf=V.mean(0)+np.einsum('ntk,ntl->tkl',diff,diff)/len(train)
  sign,logdet=np.linalg.slogdet(cf);assert np.all(sign>0);kl=.5*(np.trace(cf,axis1=-2,axis2=-1)+(mf*mf).sum(-1)-3-logdet)
  artifact=ART/'training_objective_audit_v1';artifact.mkdir(exist_ok=True)
  ap=artifact/(run+'.npz');assert not ap.exists();np.savez_compressed(ap,A=A,full_A_gradient=gfull,batch_A_gradients=gs,batch_losses=losses,batch_indices=indices,recognition_mixture_mean_by_time=mf,recognition_mixture_cov_by_time=cf,moment_matched_KL_by_time=kl)
  cosine=float(np.vdot(gmean,gfull)/(np.linalg.norm(gmean)*np.linalg.norm(gfull)))
  record={'run':run,'checkpoint_sha256':before,'source_sha256':sha(__file__),'protocol_sha256':sha(ROOT/'TRAIN_ONLY_OBJECTIVE_AUDIT54_V1.md'),'artifact':str(ap),'artifact_sha256':sha(ap),'full_bank_size':200,'native_batch_size':32,'batch_repetitions':32,'full_loss_minus_logB':full_loss-np.log(200),'mean_batch_loss_minus_logB':float(losses.mean()-np.log(32)),'gradient_cosine_mean_batch_vs_full':cosine,'full_A_gradient_norm':float(np.linalg.norm(gfull)),'mean_batch_A_gradient_norm':float(np.linalg.norm(gmean)),'gradient_mean_standard_error_frobenius_norm':float(np.linalg.norm(gse)),'full_alpha_derivative':direction,'finite_difference_alpha_derivative':fd,'finite_difference_error':fderr,'batch_alpha_derivative_mean':float(directions.mean()),'batch_alpha_derivative_MC_SE':float(directions.std(ddof=1)/np.sqrt(len(directions))),'moment_matched_recognition_KL_mean_by_time':float(kl.mean()),'moment_matched_recognition_KL_max_by_time':float(kl.max()),'mean_mixture_covariance_eigenvalues':np.linalg.eigvalsh(cf.mean(0)).tolist(),'latent_truth_read':False,'heldout_read':False,'optimizer_updates':0,'checkpoint_unchanged':sha(path)==before}
  assert record['checkpoint_unchanged'];records.append(record);dump(out/(run+'.json'),record);print(json.dumps(record),flush=True)
 dump(out/'summary.json',{'utc':utc(),'status':'COMPLETE_FROZEN_TRAINING_ONLY_AUXILIARY_OBJECTIVE_AUDIT','records':records,'runtime_seconds':time.perf_counter()-start,'source_files_unchanged':source_check(),'scope':'Same native auxiliary objective, not proper RPM likelihood. Gaussian recognition mixture moment diagnostics are not exact mixture divergence. No labels/heldout read, parameter updates or model selection.'})
if __name__=='__main__':main()
