"""Exact small joint-Gaussian check of source lag-covariance orientation; no benchmark scoring."""
import native_control_runtime
import hashlib,json,pathlib
import numpy as np
import actuated_Koopman_EM_4 as native
ROOT=pathlib.Path(__file__).resolve().parents[1]
rng=np.random.RandomState(20262791);n=16;d=2
a=np.array([[.8,.6],[-.1,.4]]);b=np.array([.05,-.02]);q=np.array([[.1,.02],[.02,.15]])
p0=np.array([[1.,.1],[.1,.7]]);m0=np.array([.1,-.3]);c=np.array([[1.,.3]]);r=.2
joint=np.block([[p0,p0@a.T],[a@p0,a@p0@a.T+q]])
obs=np.zeros((2,4));obs[0,:2]=c;obs[1,2:]=c
prior=np.r_[m0,a@m0+b];innovation=obs@joint@obs.T+r*np.eye(2)
gain=np.linalg.solve(innovation,(joint@obs.T).T).T
cov=joint-gain@obs@joint
y=rng.normal(size=(n,2));means=(prior[None]+(y-prior@obs.T)@gain.T).reshape(n,2,2)
covs=np.repeat(np.array([[cov[:2,:2],cov[2:,2:]]]),n,axis=0)
cross=np.repeat(cov[None,None,:2,2:],n,axis=0)
u=np.ones((n,1,1));evolution=np.eye(3);evolution[1:,1:]=a;evolution[1:,0]=b
filter_gain=p0@c.T/(float(c@p0@c.T)+r);filtered_cov=p0-filter_gain@c@p0
native_cross_error=0.
for j in range(n):
 filtered_mean=m0+filter_gain[:,0]*(y[j,0]-float(c@m0))
 sm,sc,cc=native._Rauch_update.py_func(means[j,1],cov[2:,2:],filtered_mean,filtered_cov,a@filtered_cov@a.T+q,evolution)
 assert np.allclose(sm,means[j,0],atol=1e-12) and np.allclose(sc,cov[:2,:2],atol=1e-12)
 native_cross_error=max(native_cross_error,float(np.max(np.abs(cc-cov[:2,2:]))))
source_v=native._optimize_explicit_Euler_generator_matrices.py_func(u,means,covs,cross)
g=np.zeros((3,3));t=np.zeros((2,3))
for j in range(n):
 m=means[j,0];delta=means[j,1]-m
 g+=np.block([[np.ones((1,1)),m[None]],[m[:,None],cov[:2,:2]+np.outer(m,m)]])
 t+=np.column_stack((delta,cov[2:,:2]-cov[:2,:2]+np.outer(delta,m)))
correct=np.linalg.solve(g,t.T).T
def expected_quadratic(v):
 aa=np.eye(2)+v[:,1:];bb=v[:,0];total=0.
 for j in range(n):
  residual=means[j,1]-aa@means[j,0]-bb
  residual_second=cov[2:,2:]+aa@cov[:2,:2]@aa.T-aa@cov[:2,2:]-cov[2:,:2]@aa.T+np.outer(residual,residual)
  total+=np.trace(np.linalg.solve(q,residual_second))
 return total/n
source_loss=expected_quadratic(source_v[0,1:]);correct_loss=expected_quadratic(correct)
assert native_cross_error<1e-12
assert np.linalg.norm(source_v[0,1:]-correct)>1e-6 and source_loss-correct_loss>1e-6
out=dict(status='verified_source_orientation_mismatch',source_revision='ef830a6d4a6cff699492d54230fc2ad1729cab20',
 source_module_sha256=hashlib.sha256(pathlib.Path(native.__file__).read_bytes()).hexdigest(),
 seed=20262791,n_trajectories=n,n_states=d,n_times=2,native_rauch_vs_exact_joint_covariance_max_abs=native_cross_error,
 actual_cross_definition='Cov(z_k,z_k+1|all y)=J_k*P_k+1',
 source_generator_covariance_term='cross - P_k;mean outer(delta_mean,mean_k)',
 correct_generator_covariance_term='cross.T - P_k;mean outer(delta_mean,mean_k)',
 source_generator=source_v[0,1:].tolist(),closed_form_generator=correct.tolist(),
 generator_frobenius_difference=float(np.linalg.norm(source_v[0,1:]-correct)),
 source_expected_transition_quadratic=source_loss,correct_expected_transition_quadratic=correct_loss,
 improvement=source_loss-correct_loss,
 scope='Unmodified source py_func math versus exact joint-Gaussian conditioning;not a new research method or benchmark result')
path=ROOT/'outputs/cross_covariance_audit.json';path.write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2),flush=True)
