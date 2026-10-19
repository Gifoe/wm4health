"""Independent products/permutation matrices, closed forms and dense Gaussian conditioning."""
import sys,pathlib,json,hashlib,math,csv
R=pathlib.Path(__file__).resolve().parents[1];B=R.parent
sys.path.insert(0,str(B/'cycle29/scripts'));import runtime_profile
import numpy as np
import torch
from scipy.linalg import solve_discrete_lyapunov,cho_factor,cho_solve,block_diag
O=R/'outputs/controlled_v2';result=json.loads((O/'result.json').read_text());p=O/'raw_controls.npz';assert hashlib.sha256(p.read_bytes()).hexdigest()==result['raw_sha256'];z=np.load(p)
perms=z['permutations'];p2=z['p2'];pv=[]
for perm in perms:
 matrix=np.zeros((4,4));matrix[perm,np.arange(4)]=1;pv.append(matrix@p2@matrix.T)
pv=np.array(pv);assert np.max(np.abs(pv-z['view_transitions']))==0
crossrows=list(csv.DictReader((O/'crossed_view.csv').open(newline='')));maxerr=0.
assert len(crossrows)==result['cross_rows']==128*6*3
for row in crossrows:
 rep=int(row['repeat']);count=int(row['anchors']);hist=int(row['history']);mapping=z[f'cross_r{rep}_mapping'];a=z[f'cross_r{rep}_anchor_states'][:count];obs=z[f'cross_r{rep}_prefix_B'][:hist]
 allowed=np.array([all(p[int(s)]==mapping[int(s)] for s in a) for p in perms]);assert allowed.sum()==math.factorial(4-len(set(a.tolist())))
 weights=np.array([.25*np.prod(m[obs[:-1],obs[1:]]) for m in pv])*allowed;weights/=weights.sum()
 pred=sum(w*m[obs[-1]] for w,m in zip(weights,pv));trueid=int(np.flatnonzero(np.all(perms==mapping,axis=1))[0]);truth=pv[trueid,obs[-1]]
 metrics={'alignment_entropy':-sum(w*math.log(w) for w in weights if w>0),'known_alignment_Brier_excess':np.dot(pred-truth,pred-truth),'known_alignment_KL_regret':np.dot(truth,np.log(truth/pred)),'true_alignment_posterior':weights[trueid]}
 for key,value in metrics.items():err=abs(value-float(row[key]));assert err<1e-12,(rep,count,hist,key,err);maxerr=max(maxerr,err)
comp=list(csv.DictReader((O/'compensation.csv').open(newline='')));eye=np.eye(4);j=np.ones((4,4))/4
for row in comp:
 h=int(row['h']);truth=j+.8**h*(eye-j)
 if row['case']=='wrong_transition_exact_belief':pred=j+.9**h*(eye-j)
 elif row['case']=='wrong_transition_compensating_state':pred=j+(8/9)*.9**h*(eye-j)
 else:pred=truth
 for k,v in [('Brier_excess',np.sum((pred-truth)**2,axis=1).mean()),('KL_regret',np.sum(truth*np.log(truth/pred),axis=1).mean())]:err=abs(v-float(row[k]));assert err<1e-13;maxerr=max(maxerr,err)
# Dense whole-24-history posterior, finite burn prior from the Lyapunov identity.
f=np.load(R/'outputs/native_v1/fixture.npz');a=f['phi'].astype(float);c=f['F'].astype(float);d=len(a);tmax=24
lq=torch.linalg.cholesky(torch.from_numpy(f['Sigma_d'].copy())).numpy().astype(float);lr=torch.linalg.cholesky(torch.from_numpy(f['Sigma_p'].copy())).numpy().astype(float)
q=.01*lq@lq.T;rr=.09*lr@lr.T;pinf=solve_discrete_lyapunov(a,q);a100=np.linalg.matrix_power(a,100);p0=pinf+a100@(np.eye(d)-pinf)@a100.T
ps=[p0]
for _ in range(tmax-1):ps.append(a@ps[-1]@a.T+q)
statecov=np.empty((tmax*d,tmax*d))
for i in range(tmax):
 for jn in range(tmax):statecov[i*d:(i+1)*d,jn*d:(jn+1)*d]=np.linalg.matrix_power(a,i-jn)@ps[jn] if i>=jn else ps[i]@np.linalg.matrix_power(a,jn-i).T
oracleerr=0.
for rec in range(4):
 mask=f['mask_test'][rec,:,:tmax].astype(bool);flatidx=np.flatnonzero(mask.T.reshape(-1));hfull=np.kron(np.eye(tmax),c);h=hfull[flatidx];rcov=np.kron(np.eye(tmax),rr)[np.ix_(flatidx,flatidx)];yy=f['X_test_full'][rec,:,:tmax].T.reshape(-1)[flatidx]
 cov=h@statecov@h.T+rcov;fac=cho_factor(cov,lower=True);cross=statecov[-d:,:]@h.T;m=cross@cho_solve(fac,yy);pcur=ps[-1]-cross@cho_solve(fac,cross.T)
 for step in range(1,9):
  m=a@m;pcur=a@pcur@a.T+q
  if step in [1,4,8]:
   mu=c@m;vv=c@pcur@c.T+rr;err=max(np.max(np.abs(mu-z[f'oracle_r{rec}_h{step}_mean'])),np.max(np.abs(vv-z[f'oracle_r{rec}_h{step}_cov'])));assert err<1e-9,(rec,step,err);oracleerr=max(oracleerr,float(err))
out={'status':'PASS','cross_rows':len(crossrows),'independent_cross_metric_cells':len(crossrows)*4,'compensation_rows':len(comp),'max_statistic_abs_difference':maxerr,'independent_dense_gaussian_predictions':12,'max_dense_gaussian_abs_difference':oracleerr,'raw_sha256':result['raw_sha256'],'test_scored':False,'limits':'Independent exact controls, not neural method evaluation; native GPU training not repeated.'}
(O/'verification_v2.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
