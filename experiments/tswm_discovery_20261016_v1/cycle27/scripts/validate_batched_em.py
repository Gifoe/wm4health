"""Validate source-mode backend against unmodified native100 checkpoint before changing execution."""
import native_em_positive_control as common
import batched_em as batch
import json,pathlib,time
import numpy as np
import actuated_Koopman_EM_4 as native
ROOT=common.ROOT;SOURCE=common.OUT;OUT=ROOT/'outputs/backend_validation'
def main():
 assert not OUT.exists();OUT.mkdir();started=time.monotonic()
 data=np.load(SOURCE/'data_train_continuation.npz');y=np.ascontiguousarray(data['y'][:,:250]);u=np.ascontiguousarray(data['u'][:,:249]);scaled=.01*u
 p=batch.initial(20262711,u,common.reflect_eigs);native_initial=np.load(SOURCE/'seed20262711/weights_init.npz')
 for key in common.PARAMETERS:assert np.allclose(p[key],native_initial[key],rtol=0,atol=0),key
 m,s,c,ll=batch.estep(p,y,scaled,implicit=True)
 # Source moment functions evaluated directly,without JIT/backend ambiguity.
 v=native._optimize_explicit_Euler_generator_matrices.py_func(scaled,m,s,c)
 updated=batch.mstep(p,y,scaled,m,s,c,False)
 assert np.allclose(v,updated['V_matrices'],atol=2e-7,rtol=2e-6)
 full=np.eye(5)+np.einsum('nti,ijk->ntjk',scaled,v)
 q=native._optimize_process_noise_from_A.py_func(full,m,s,c)
 assert np.allclose(q,updated['Sig_v'],atol=2e-7,rtol=2e-6)
 one_step=dict(generator_max_abs=float(np.max(np.abs(v-updated['V_matrices']))),noise_max_abs=float(np.max(np.abs(q-updated['Sig_v']))))
 for iteration in range(1,101):
  p=batch.mstep(p,y,scaled,m,s,c,False);m,s,c,ll=batch.estep(p,y,scaled)
  if iteration%20==0:common.progress(phase='backend_validation',iteration=iteration,likelihood=ll,elapsed_s=time.monotonic()-started)
 reference=np.load(SOURCE/'seed20262711/weights_iter100.npz');errors={}
 for key in common.PARAMETERS:
  errors[key]=float(np.max(np.abs(p[key]-reference[key])));assert np.allclose(p[key],reference[key],atol=2e-7,rtol=2e-6),(key,errors[key])
 likelihood_error=abs(ll-float(reference['likelihood']));assert likelihood_error<1e-4
 np.savez_compressed(OUT/'source_mode100.npz',**p,likelihood=np.array(ll),iteration=np.array(100))
 result=dict(status='passed',one_step=one_step,iteration100_max_abs=errors,likelihood_abs_error=likelihood_error,runtime_s=time.monotonic()-started,
    source_checkpoint_sha256=common.digest(SOURCE/'seed20262711/weights_iter100.npz'),backend_sha256=common.digest(pathlib.Path(batch.__file__)),
    scope='Source equations including covariance orientation issue faithfully reproduced;no held-out scores')
 common.write(OUT/'results.json',result);print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
