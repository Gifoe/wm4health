from common87_v1 import *
from physics87_v1 import *
from scipy.integrate import solve_ivp
import time
def main():
 start=time.perf_counter();m=read(O/'DATA_MANIFEST87_V1.json');checks=[];scalars=0;worst=0.;max_scaled=0.
 for row in m['files']:
  assert sha(row['array_path'])==row['array_sha256']
  if 'arm' in row:continue
  with np.load(row['array_path']) as z:d={k:z[k] for k in z.files}
  fam=row['family'];fi=FAMILIES.index(fam);rng=np.random.default_rng(np.random.SeedSequence([row['seed'],fi]));p=parameters(fam,row['pool_count'],rng);ic=rng.normal(0,.8,(len(p),9,2));assert np.array_equal(p,d['parameters']) and np.array_equal(ic,d['initial'])
  maxima=abs(d['paths']).max(axis=(1,2,3));assert np.array_equal(maxima<2.5,d['accepted']);assert int(d['accepted'].sum())==row['accepted_count']
  srng=np.random.default_rng(20268790+fi);pick=np.unique(np.r_[srng.choice(len(p),23,replace=False),np.argsort(maxima)[-2:]])
  for k in pick:
   def rhs(t,y):return field(fam,p[k:k+1],y.reshape(1,9,2)).ravel()
   sol=solve_ivp(rhs,(0,10),ic[k].ravel(),method='DOP853',t_eval=d['times'],rtol=2e-11,atol=2e-12);assert sol.success and sol.y.shape==(18,201)
   reference=sol.y.T.reshape(201,9,2).transpose(1,0,2);err=abs(reference-d['paths'][k]);absmax=float(err.max());scaled=float((err/np.maximum(abs(reference),1)).max());worst=max(worst,absmax);max_scaled=max(max_scaled,scaled);scalars+=reference.size
   assert scaled<.002,(fam,row['split'],int(k),absmax,scaled)
  A=read_arm(row['split'],fam,'whole_filter');B=read_arm(row['split'],fam,'resampled_ic');U=read_arm(row['split'],fam,'unselected');n=len(A['mechanism_ids'])
  assert np.array_equal(A['mechanism_ids'],np.flatnonzero(d['accepted'])[:n]);assert np.array_equal(B['mechanism_ids'],A['mechanism_ids']);assert np.array_equal(U['mechanism_ids'],np.arange(n))
  for key in ['parameters','locations','drift','noise']:assert np.array_equal(A[key],B[key]),key
  assert np.array_equal(A['context_clean'],d['paths'][A['mechanism_ids'],:2,:33]);assert np.array_equal(U['context_clean'],d['paths'][:n,:2,:33])
  for arm in [A,B,U]:
   assert np.array_equal(arm['context'],arm['context_clean']+arm['noise']);assert np.allclose(arm['drift'],field(fam,arm['parameters'],arm['locations']),atol=1e-4,rtol=1e-6)
  checks.append(dict(family=fam,split=row['split'],n=row['pool_count'],accepted=row['accepted_count'],independent_reference_mechanisms=len(pick),independent_reference_paths=len(pick)*9))
 write(O/'DATA_VERIFICATION87_V1.json',dict(utc=utc(),status='PASS_PARAMETER_REPLAY_PAIRED_DATA_AND_INDEPENDENT_DOP853',checks=checks,reference_scalars=scalars,max_abs_dop853_rk4=worst,max_scaled_abs_dop853_rk4=max_scaled,eval_opened=False,manifest_sha256=sha(O/'DATA_MANIFEST87_V1.json'),source_sha256=sha(__file__),seconds=time.perf_counter()-start));log(dict(stage='independent_data_verification',reference_scalars=scalars,max_abs=worst,max_scaled=max_scaled))
def read_arm(split,fam,arm):
 with np.load(RAW/'population_v1'/f'{split}_{fam}_{arm}_v1.npz') as z:return {k:z[k] for k in z.files}
if __name__=='__main__':main()
