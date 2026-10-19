"""Independent NumPy/SciPy likelihood, path metrics and count/state cross moments."""
import pathlib,sys
R=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(R.parent/'cycle29/scripts'));import runtime_profile
import numpy as np,json,hashlib,math,datetime
from scipy.special import logsumexp,gammaln
from scipy.spatial.distance import pdist
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8'))
def main():
 out=R/'outputs/eval_v1';assert not (out/'verification.json').exists();raw_manifest=read(out/'raw_manifest.json');summary=read(out/'summary.json');bundles={};count=0;maxes=0.;maxmse=0.;maxll=0.;maxcov=0.;group_values={}
 for ent in read(out/'model_bundles.json'):
  p=out/ent['path'];assert sha(p)==ent['sha256'];bundles[ent['model']]={k:v.astype('float64') if k!='mean_clock' else bool(v) for k,v in np.load(p).items()}
 for ent in raw_manifest:
  p=out/ent['path'];assert sha(p)==ent['sha256'];raw=np.load(p);bundle=bundles[ent['model']];P=bundle['P'];mark=bundle['mark'];prior=bundle['prior'];rate=bundle['rate'];initial=raw['initial'];counts=raw['counts'].astype('float64');rho=float(ent['rho']);truth=raw['truth'].astype('float64');assert np.array_equal(raw['truth'],np.load(R/'data'/ent['case']/'sealed_eval_auditor_truth.npz')['scheduled_future']);observations=np.load(R/'data'/ent['case']/'sealed_eval_observations.npz');policy={.25:0,1.:1,2.:2}[rho];assert all(np.array_equal(raw[k],observations[f'policy{policy}_{k}']) for k in ['initial','counts','rho']);assert raw['draws'].shape==(256,64,32) and np.isfinite(raw['draws']).all();logalpha=np.log(prior)[None]+np.log(mark[:,initial].T);ll=logsumexp(logalpha,axis=-1);logalpha-=ll[:,None];logP=np.log(P)
  for t in range(64):
   c=counts[:,t];N=c.sum(-1);rates=bundle['mean_rates'][initial,t,None] if bundle['mean_clock'] else rate[None];emission=c@np.log(mark).T+N[:,None]*np.log(.125*rho*rates)-.125*rho*rates-gammaln(c+1).sum(-1)[:,None];logalpha+=emission;scale=logsumexp(logalpha,axis=-1);ll+=scale;logalpha-=scale[:,None];logalpha=logsumexp(logalpha[:,:,None]+logP[None],axis=1)
  nll=-ll/65;err=float(np.max(abs(nll-raw['nll_per65'])));maxll=max(maxll,err);assert err<1e-4,(ent['path'],'likelihood',err);scores=[]
  for i,paths in enumerate(raw['draws'].astype('float64')):
   es=(np.linalg.norm(paths-truth[i],axis=1).mean()-pdist(paths).mean()/2)/math.sqrt(32);mse=np.mean((paths.mean(0)-truth[i])**2);bounds=np.percentile(paths,[5,95],axis=0);coverage=float(np.logical_and(truth[i]>=bounds[0],truth[i]<=bounds[1]).mean());maxes=max(maxes,abs(es-float(raw['energy'][i])));maxmse=max(maxmse,abs(mse-float(raw['mse'][i])));assert abs(es-float(raw['energy'][i]))<3e-6 and np.isclose(mse,raw['mse'][i],atol=3e-6,rtol=3e-6) and coverage==float(raw['coverage'][i]);scores.append((es,mse,coverage,nll[i]));count+=1
  group_values[(ent['model'],rho)]=np.mean(scores,0)
 for g in summary['groups']:assert np.allclose(group_values[(g['model'],g['rho'])],[g[m] for m in ['energy','mse','coverage90','nll_per65']],atol=1e-4,rtol=1e-5)
 # Independent pair-moment sum, rather than the running-count recursion.
 for row in read(out/'coupling.json'):
  b=bundles[row['model']];P=b['P'];grid=b['grid'];mark=b['mark'];prior=b['prior'];alpha=(prior[:,None]*mark).T;prob=alpha.sum(-1);alpha/=prob[:,None];initial=alpha.copy();EZ=(initial@np.linalg.matrix_power(P,32))@grid;cov=np.zeros(8)
  if not b['mean_clock']:
   for t in range(32):future=np.linalg.matrix_power(P,32-t)@grid;cov+=.125*row['rho']*((alpha*(b['rate']*future)[None]).sum(-1)-(alpha@b['rate'])*EZ);alpha=alpha@P
  err=float(np.max(abs(cov-np.array(row['conditional_count_state_covariance']))));maxcov=max(maxcov,err);assert err<2e-5,(row['model'],row['rho'],err)
 assert len(raw_manifest)==54 and count==13824 and summary['unique_physical_EVAL_trajectories']==512
 schedules={}
 for p in (R/'outputs/fits').iterdir():
  fit=read(p/'fit_summary.json');cfg=read(p/'config.json');assert sha(p/'selected.pt')==fit['selected_sha256'] and sha(p/'last.pt')==fit['last_sha256'];assert sha(p/'schedule.npz')==fit['schedule_sha256'];schedules.setdefault(cfg['seed'],set()).add(fit['schedule_sha256']);assert cfg['common_sha256']==sha(R/'scripts/common47.py')
 assert len(schedules)==2 and all(len(s)==1 for s in schedules.values());result={'status':'PASS_FROZEN_FINITE_GRID_SCREEN','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'raw_groups':54,'model_origin_policy_entries':count,'unique_physical_EVAL_trajectories':512,'max_energy_arithmetic_error':maxes,'max_mse_arithmetic_error':maxmse,'max_nll_per65_independent_error':maxll,'max_covariance_independent_error':maxcov,'per_seed_schedules_identical':True,'summary_sha256':sha(out/'summary.json'),'manifest_sha256':sha(out/'raw_manifest.json'),'verifier_source_sha256':sha(__file__),'not_continuous_or_real_domain_validation':True};(out/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
