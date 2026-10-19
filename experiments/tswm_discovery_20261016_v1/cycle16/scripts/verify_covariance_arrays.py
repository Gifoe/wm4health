"""Independent NumPy closed-form verification of saved diagnostic arrays."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import argparse,hashlib,json,pathlib
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def full(x):
 d=len(x)//3;P=np.zeros((2*d,2*d));idx=np.arange(d)
 P[idx,idx]=x[:d];P[idx+d,idx+d]=x[d:2*d];P[idx,idx+d]=P[idx+d,idx]=x[2*d:];return P
def pack(P):
 d=len(P)//2;return np.r_[np.diag(P[:d,:d]),np.diag(P[d:,d:]),np.diag(P[:d,d:])]
def main(directory):
 root=pathlib.Path(directory);result=json.loads((root/'results.json').read_text());assert sha(root/'arrays.npz')==result['arrays_sha256']
 with np.load(root/'arrays.npz',allow_pickle=False) as f:a={k:f[k] for k in f.files}
 A,C,Q=a['A'],a['C'],a['Q'];H=a['native_mean'].shape[1]
 errors=[];min_eigenvalue=float('inf')
 for b in range(len(a['origins'])):
  P0=full(a['initial_fast_covariance_summary'][b].astype(np.float64));S=full(a['context_covariance_summary'][b].astype(np.float64))
  min_eigenvalue=min(min_eigenvalue,float(np.linalg.eigvalsh(P0).min()),float(np.linalg.eigvalsh(S).min()))
  power=np.eye(len(A));G=np.zeros_like(A);noise_sum=np.zeros_like(A);independent_noise_sum=np.zeros_like(A)
  for h in range(H):
   G+=power@C;noise_sum+=power@Q@power.T;independent_noise_sum+=power@(C@S@C.T+Q)@power.T
   power=A@power;initial=power@P0@power.T
   joint=initial+G@S@G.T+noise_sum;independent=initial+independent_noise_sum
   for name,value in [('dense_joint',joint),('dense_independent',independent)]:
    stored=a[name+'_latent_covariance_summary'][b,h];want=pack(value)
    errors.append(float(np.max(np.abs(want-stored))))
    np.testing.assert_allclose(want,stored,rtol=2e-10,atol=2e-10)
 metric_errors=[]
 for row in result['rows']:
  variance=a[row['control']+'_variance'].astype(np.float64);mean=a['native_mean'].astype(np.float64);target=a['target'].astype(np.float64)
  err=target-mean
  values={'NLL':float(np.mean(.5*(np.log(2*np.pi*variance)+err**2/variance))),
          'MAE_standardized':float(np.mean(np.abs(err))),'RMSE_standardized':float(np.sqrt(np.mean(err**2))),
          'coverage95':float(np.mean(np.abs(err)<=1.959963984540054*np.sqrt(variance))),
          'mean_predicted_variance':float(np.mean(variance))}
  for k,v in values.items():
   metric_errors.append(abs(v-row[k]));assert np.isclose(v,row[k],rtol=1e-12,atol=1e-12),(k,v,row[k])
 assert min_eigenvalue>=-1e-10
 receipt={'passed':True,'independent_algorithm':'Closed-form powers and sums of A; no reuse of recursive diagnostic implementation',
          'origins_verified':len(a['origins']),'horizons_verified':H,'dense_trajectories_verified':2,
          'maximum_closed_form_covariance_error':max(errors),'maximum_metric_error':max(metric_errors),
          'minimum_initial_factor_covariance_eigenvalue':min_eigenvalue,
          'results_sha256':sha(root/'results.json'),'arrays_sha256':sha(root/'arrays.npz'),'verifier_sha256':sha(pathlib.Path(__file__))}
 out=root/'independent_verification.json';assert not out.exists();out.write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps(receipt))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('directory');args=p.parse_args();main(args.directory)
