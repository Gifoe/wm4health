from models65 import *
from scipy.stats import wasserstein_distance
torch.set_num_threads(4);pp=paths();records=[];raw={};N=8192
def plan(kind):return np.zeros(H) if kind=='zero' else .4*np.sin(2*np.pi*(np.arange(H)+1)/8)
def sample_do(n,mech,u,rng):
 y=np.empty((n,H));last=np.full(n,.25)
 for t in range(H):last=drift(last,u[t],mech)+sigma(mech)*rng.normal(size=n);y[:,t]=last
 return y
def sample_obs(n,mech,u,rng,cap=2**25):
 out=[];kept=0;proposals=0;accepted=0
 while kept<n and proposals<cap:
  b=min(32768,cap-proposals);x=sample_do(b,mech,u,rng);logacc=-.5*np.sum(((u[1:]+.75*x[:,:-1])/.5)**2,axis=1);take=x[np.log(rng.random(b))<logacc];proposals+=b;accepted+=len(take);out.append(take[:n-kept]);kept+=min(len(take),n-kept)
 assert kept==n,('REJECTION_ORACLE_INCOMPLETE',mech,proposals,kept)
 return np.concatenate(out),dict(proposed=proposals,accepted_candidates=accepted,retained=n,acceptance_rate=accepted/proposals)
rng=np.random.default_rng(650999);D=rng.normal(size=(128,H));D=D/np.linalg.norm(D,axis=1,keepdims=True);D=np.r_[D,np.eye(H)];raw['directions']=D
for mi,mech in enumerate(MECHANISMS):
 for pi,pk in enumerate(['zero','sine']):
  state('oracle_state65_v1.json',dict(status='GENERATING_EXACT_REFERENCE',mechanism=mech,query=pk));u=plan(pk);rng=np.random.default_rng(np.random.SeedSequence([65010,mi,pi]));xd=sample_do(N,mech,u,rng);xo,receipt=sample_obs(N,mech,u,rng);key=mech+'_'+pk;raw[key+'_do']=xd;raw[key+'_obs']=xo;checks={}
  if mech=='linear':
   L=np.eye(H)-.8*np.eye(H,k=-1);b=.7*u;b[0]+=.8*.25;Q=L.T@L/.25;hd=L.T@b/.25;P=np.diag(np.r_[np.repeat(.75**2/.25,H-1),0.]);ell=np.r_[-.75*u[1:]/.25,0.]
   for name,x,QQ,hh in [('do',xd,Q,hd),('obs',xo,Q+P,hd+ell)]:
    C=np.linalg.inv(QQ);m=C@hh;mean_z=float(np.max(np.abs(x.mean(0)-m)/np.sqrt(np.diag(C)/N)));sampleC=np.cov(x,rowvar=False,ddof=0);Cse=np.sqrt((np.outer(np.diag(C),np.diag(C))+C*C)/N);cov_z=float(np.max(np.abs(sampleC-C)/Cse));assert mean_z<6 and cov_z<6,(name,mean_z,cov_z);checks[name]=dict(mean_max_standard_errors=mean_z,covariance_max_standard_errors=cov_z)
  floors=[]
  for seed in SEEDS:
   rng=np.random.default_rng(np.random.SeedSequence([65040,mi,pi,seed]));raw[key+f'_iid_do_s{seed}']=sample_do(512,mech,u,rng);xx,rr=sample_obs(512,mech,u,rng);raw[key+f'_iid_obs_s{seed}']=xx;floors.append(dict(seed=seed,observational_rejection=rr))
  records.append(dict(mechanism=mech,query=pk,reference_records=N,observational_rejection=receipt,linear_analytic_checks=checks,iid_floor_receipts=floors));print(json.dumps(records[-1]),flush=True)
p=pp['artifact_root']/'ORACLE_REFERENCE65_V1.npz';np.savez_compressed(p,**raw);dump(OUT/'ORACLE65_V1.json',dict(utc=utc(),status='EXACT_ORACLES_QUALIFIED',records=records,artifact=str(p),artifact_sha256=sha(p),directions=136,reference_records_per_law=N,MC_floor_records_per_law=512,scope='Known-SCM do simulation and exact rejection observational conditionals, not learned model performance',evaluation_spec_sha256=sha(ROOT/'EVALUATION_SPEC65_V1.md')));state('oracle_state65_v1.json',dict(status='EXACT_ORACLES_QUALIFIED',queries=4,artifact_sha256=sha(p)))
