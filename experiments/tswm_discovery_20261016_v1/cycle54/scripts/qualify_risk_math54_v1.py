"""Only analytic/dummy fixtures; no C54 data, model or heldout access."""
import datetime,hashlib,json,pathlib
import numpy as np
from risk_math54_v1 import decompose,paired_interval,percentile,ratio
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def main():
    target=OUT/'QUALIFICATION_RISK_MATH54_V1.json';assert not target.exists()
    rng=np.random.default_rng(5490);pred=rng.normal(size=(9,6,3));mu=rng.normal(size=(9,6,3));truth=rng.normal(size=(9,6,3));_,identity=decompose(pred,mu,truth)
    K=3;L=np.array([[.7,0,0],[.2,.4,0],[-.1,.3,.5]]);noise=np.concatenate([np.eye(K),-np.eye(K)])@L.T*np.sqrt(K)
    means=mu[:2];p=pred[:2];expanded_mu=np.repeat(means,2*K,axis=0);expanded_p=np.repeat(p,2*K,axis=0);z=(means[:,None]+noise[None,:,None,:]).reshape(-1,6,3)
    d,cubature_identity=decompose(expanded_p,expanded_mu,z)
    cross=float(np.max(abs(d['finite_sample_cross'].reshape(2,2*K,6).mean(1))));known=float(np.trace(L@L.T)/K)
    noise_error=float(np.max(abs(d['oracle_noise'].reshape(2,2*K,6).mean(1)-known)))
    assert cross<1e-12 and noise_error<1e-12
    ids=rng.integers(0,9,size=(2048,9));ci=paired_interval(np.full((9,6),.25),np.full((9,6),.5),ids)
    assert all(r['pointwise95_interval']==[.25,.25] and r['normalized_pointwise95_interval']==[.5,.5] for r in ci)
    assert ratio(1.,0.) is None and all(r['normalized_delta'] is None for r in paired_interval(np.ones((9,6)),np.zeros((9,6)),ids))
    values=rng.normal(size=2048);qe=max(abs(percentile(values,q)-float(np.quantile(values,q))) for q in [.025,.5,.975]);assert qe<1e-12
    result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'PASS_INDEPENDENT_DUMMY_BAYES_RISK_AND_SEQUENCE_BOOTSTRAP_ARITHMETIC','dummy_identity_error':identity,'cubature_identity_error':cubature_identity,'symmetric_cubature_cross_term_error':cross,'known_trace_covariance_error':noise_error,'explicit_percentile_vs_numpy_error':qe,'constant_paired_difference_check':True,'zero_energy_undefined_check':True,'formal_model_reads':0,'formal_data_or_label_reads':0,'optimizer_updates':0,'module_sha256':sha(ROOT/'scripts/risk_math54_v1.py'),'qualification_script_sha256':sha(__file__)}
    target.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
