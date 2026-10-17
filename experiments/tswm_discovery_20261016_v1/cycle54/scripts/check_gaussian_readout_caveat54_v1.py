"""An exact-oracle counterexample to interpreting a point-map Gaussian as calibration."""
import datetime,hashlib,json,pathlib
import numpy as np
from gaussian_belief_math54_v1 import scores
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
paths=[OUT/'gaussian_belief_audit_v1/summary.json',OUT/'evaluation54.json',OUT/'em_option_learning_v1/evaluation/summary.json'];before={p:sha(p) for p in paths}
g=read(paths[0]);assert g['status']=='COMPLETE_ALL16_FIXED_HORIZON_GAUSSIAN_BELIEF_AUDIT'
# Z~N(0,I), Y=Z1+N(0,.25): the exact posterior has an unobserved unit-variance axis.
posterior_mean=np.array([[-.8,0.],[0.,0.],[.8,0.]])
posterior_cov=np.broadcast_to(np.diag([.2,1.]),(3,2,2)).copy();truth=np.array([[-.4,-.6],[.5,.7],[.9,-.3]])
population_mean_cov=np.diag([.8,0.]);population_cross_cov=np.diag([.8,0.]);W=np.linalg.pinv(population_mean_cov)@population_cross_cov
assert np.array_equal(W,np.diag([1.,0.]))
bad=scores(posterior_mean@W,W.T@posterior_cov@W,posterior_mean,posterior_cov,truth,5.991464547107979)
assert not np.any(bad['valid_covariance'])
identity=scores(posterior_mean,posterior_cov,posterior_mean,posterior_cov,truth,5.991464547107979)
assert np.max(abs(identity['gaussian_kl']))<1e-12
near=[]
for epsilon in [.1,.01,.001]:
    V=np.diag([1.,epsilon]);r=scores(posterior_mean@V,V.T@posterior_cov@V,posterior_mean,posterior_cov,truth,5.991464547107979)
    expected=.5*(epsilon**-2-1+2*np.log(epsilon))
    assert np.max(abs(r['gaussian_kl']-expected))<1e-8
    near.append({'second_axis_scale':epsilon,'mapped_oracle_kl':float(r['gaussian_kl'].mean()),'closed_form_kl':float(expected)})
conditions=[{k:c[k] for k in ['run','state_map_rank','state_map_condition']} for c in g['native_checks']]
record={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'EXACT_ORACLE_POINT_READOUT_COUNTEREXAMPLE_CONFIRMED','fixture':'Correct Gaussian posterior, one physically unobserved state component, no learned model or formal signal reads','population_OLS_map':W.tolist(),'identity_gauge_KL':float(identity['gaussian_kl'].max()),'point_map_all_covariances_singular':True,'nearly_singular_examples':near,'existing_C54_map_conditions':conditions,'interpretation':'The registered mapped-surrogate scores remain numerically valid under their stated map. Their large magnitude alone cannot identify native uncertainty error: information loss or a shrinking point readout can manufacture overconfidence even from an exact oracle. Condition alone does not attribute the actual C54 errors.','optimizer_updates':0,'formal_model_or_signal_reads':0,'reads_existing_scoring_reports_only':True,'previous_scoring_hashes_unchanged':{p.relative_to(ROOT).as_posix():v for p,v in before.items()},'script_sha256':sha(__file__),'accepted_final_directions':0}
for p,value in before.items():assert sha(p)==value
target=OUT/'gaussian_readout_caveat_v1.json'
with target.open('x',encoding='utf-8') as f:json.dump(record,f,indent=2)
text='# Interpretation guard for the mapped Gaussian scores\n\nThis post-scoring interpretation audit preserves all registered C54 scores and all16 models. It introduces no method, recalibration or additional training.\n\n'
text+='A verified exact-oracle example prevents a misleading conclusion. Let Z have standard Gaussian prior and observe only Z1 with Gaussian noise variance.25. Its exact posterior covariance is diag(.2,1). The population minimum-norm OLS map from its posterior mean to Z is diag(1,0), since the second posterior mean is always zero. Pushing the posterior covariance through that point map makes it singular, even though the original posterior is exactly calibrated. An identity state gauge retains KL0. Near-zero readout scales.1,.01,.001 instead yield finite KL values below without changing the oracle model:\n\n| Axis scale | Artificial mapped KL |\n|---:|---:|\n'
for r in near:text+=f"|{r['second_axis_scale']}|{r['mapped_oracle_kl']:.9g}|\n"
text+='\nTherefore, outputs/gaussian_belief_audit_v1/ remains a diagnostic of the declared mapped Gaussian surrogate. Its large values cannot by themselves support a native-posterior calibration failure or motivate a new uncertainty framework. The point mapping was supervised for mean recovery, and no faithful deterministic physical-coordinate correspondence was established. The small-task default maps have modest condition numbers, whereas the medium-task existing-True maps have conditions about125 and399; these checks flag interpretation risk but do not quantify the actual mapping contribution.\n\nThe state-mean forecast comparisons and oracle-risk decomposition remain unchanged. A future uncertainty comparison needs an established latent-state gauge or an observation-space generative readout, with ordinary mapping and inference controls. Do not select a favorable replacement on these heldout results. No new state map is fitted here, and this oracle fixture is a methodological safeguard rather than experimental evidence of a research gap. Machine-readable equations, conditions and immutable source-result hashes are in outputs/gaussian_readout_caveat_v1.json.\n'
p=ROOT/'GAUSSIAN_READOUT_CAVEAT54_V1.md';assert not p.exists();p.write_text(text,encoding='utf-8')
print(json.dumps({'status':record['status'],'oracle_kl':record['identity_gauge_KL'],'near_singular_kl':near,'report_sha256':sha(p)}),flush=True)
