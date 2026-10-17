"""Independent density/integration and coordinate checks on dummy Gaussians only."""
import datetime, hashlib, json, pathlib
import numpy as np
from scipy.stats import multivariate_normal, chi2
from gaussian_belief_math54_v1 import scores
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs'
rng=np.random.default_rng(54881);K=3;N=7
mm=rng.normal(size=(N,K));mo=rng.normal(size=(N,K));truth=rng.normal(size=(N,K))
u=rng.normal(size=(N,K,K));v=rng.normal(size=(N,K,K));cm=u@u.swapaxes(-1,-2)+.3*np.eye(K);co=v@v.swapaxes(-1,-2)+.5*np.eye(K)
s=scores(mm,cm,mo,co,truth,chi2.ppf(.95,K))
independent_model=np.array([-multivariate_normal.logpdf(truth[i],mean=mm[i],cov=cm[i]) for i in range(N)])
independent_oracle=np.array([-multivariate_normal.logpdf(truth[i],mean=mo[i],cov=co[i]) for i in range(N)])
density_error=max(float(np.max(abs(s['learned_nll']-independent_model))),float(np.max(abs(s['oracle_nll']-independent_oracle))))
integration=[]
for i in range(N):
    root=np.linalg.cholesky(co[i]);points=np.concatenate([mo[i]+np.sqrt(K)*root.T,mo[i]-np.sqrt(K)*root.T])
    excess=multivariate_normal.logpdf(points,mean=mo[i],cov=co[i])-multivariate_normal.logpdf(points,mean=mm[i],cov=cm[i])
    integration.append(float(excess.mean()))
integration_error=float(np.max(abs(np.array(integration)-s['gaussian_kl'])))
T=np.array([[1.2,.3,-.1],[.2,.9,.1],[0.,-.2,1.5]]);offset=np.array([.2,-.1,.3]);_,logdet=np.linalg.slogdet(T)
st=scores(mm@T.T+offset,T@cm@T.T,mo@T.T+offset,T@co@T.T,truth@T.T+offset,chi2.ppf(.95,K))
invariance_error=max(float(np.max(abs(st[k]-s[k]))) for k in ['gaussian_kl','mean_kl','covariance_kl','realized_nll_excess','mahalanobis_squared'])
jacobian_error=float(np.max(abs(st['learned_nll']-s['learned_nll']-logdet)))
same=scores(mo,co,mo,co,truth,chi2.ppf(.95,K));same_error=float(np.max(abs(same['gaussian_kl'])))
singular=cm.copy();singular[0]=0
invalid=scores(mm,singular,mo,co,truth,chi2.ppf(.95,K))
assert not invalid['valid_covariance'][0] and np.isnan(invalid['gaussian_kl'][0]) and np.sum(invalid['valid_covariance'])==N-1
assert max(density_error,integration_error,invariance_error,jacobian_error,same_error)<1e-10
record={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'PASS_INDEPENDENT_DUMMY_GAUSSIAN_DENSITY_INTEGRATION_AND_AFFINE_INVARIANCE','scipy_density_max_error':density_error,'sigma_point_expected_log_score_error':integration_error,'affine_invariance_max_error':invariance_error,'nll_jacobian_error':jacobian_error,'identical_gaussian_kl_error':same_error,'singular_covariance_not_repaired':True,'formal_data_or_model_reads':0,'optimizer_updates':0,'math_script_sha256':hashlib.sha256((ROOT/'scripts/gaussian_belief_math54_v1.py').read_bytes()).hexdigest()}
p=OUT/'QUALIFICATION_GAUSSIAN_BELIEF54_V1.json'
with p.open('x',encoding='utf-8') as f:json.dump(record,f,indent=2)
print(json.dumps(record),flush=True)
