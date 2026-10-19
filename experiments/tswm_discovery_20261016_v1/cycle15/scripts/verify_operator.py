"""Independent arithmetic from saved arrays, with simultaneous MC intervals."""
import runtime_profile
import hashlib,json,math,pathlib,datetime
import numpy as np
from scipy.stats import norm,chi2
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 r=json.loads((OUT/'operator/results.json').read_text());cost=json.loads((OUT/'cost/results.json').read_text())
 max_native=0.;max_joint=0.;negative=0;checks=0
 with np.load(OUT/'operator/variance_sequences.npz',allow_pickle=False) as arrays:
  for row in r['rows']:
   a,c,s,h=row['a'],row['c'],row['context_variance'],row['horizon'];powers=np.power(a,np.arange(h,dtype=np.float64))
   p=(a**h)**2*.05+(c*powers.sum())**2*s+row['process_variance']*np.dot(powers,powers)
   f=(a**h)**2*.05+(c*c*s+row['process_variance'])*np.dot(powers,powers)
   key=f"{row['native_branch']}_a{a}_c{c}_s{s}"
   vn=float(arrays[key+'_native'][h-1]);vj=float(arrays[key+'_joint'][h-1])
   max_native=max(max_native,abs(vn-f));max_joint=max(max_joint,abs(vj-p))
   assert abs(vn-row['native_variance'])<1e-10 and abs(vj-row['augmented_state_variance'])<1e-10
   np.testing.assert_allclose(row['persistent_nominal95_actual_marginal_coverage'],norm.cdf(norm.ppf(.975)*math.sqrt(f/p))*2-1,atol=1e-10)
   if h==1 or a==0 or c==0 or s==0:
    assert abs(p-f)<1e-8;negative+=1
   checks+=1
 assert max_native<1e-8 and max_joint<1e-8 and checks==630
 mc=[];alpha=.05/(len(r['monte_carlo'])*2*2)
 for index in range(6):
  path=OUT/f'operator/mc_case{index}.npz'
  with np.load(path,allow_pickle=False) as arrays:
   assert arrays['persistent'].shape==arrays['fresh'].shape==(7,50000,2)
   assert all(np.isfinite(arrays[key]).all() for key in arrays.files)
   for j,h in enumerate(arrays['horizons']):
    expected=next(x for x in r['monte_carlo'] if x['case']==index and x['horizon']==h)
    for family in ['persistent','fresh']:
     values=arrays[family][j].astype(np.float64);v=values.var(0,ddof=1);target=expected[family+'_expected_variance']
     np.testing.assert_allclose(v,expected[family+'_empirical_variances'],rtol=2e-7,atol=2e-7)
     lo=(len(values)-1)*v/chi2.ppf(1-alpha/2,len(values)-1);hi=(len(values)-1)*v/chi2.ppf(alpha/2,len(values)-1)
     mc.append({'case':index,'horizon':int(h),'family':family,'variance_CI_lower':lo.tolist(),'variance_CI_upper':hi.tolist(),
                'expected_variance':target,'expected_inside_CI':((lo<=target)&(target<=hi)).tolist()})
    vp=expected['persistent_expected_variance'];vf=expected['fresh_expected_variance'];x=arrays['persistent'][j].astype(np.float64)
    covp=(np.abs(x)<=norm.ppf(.975)*math.sqrt(vp)).mean(0);covn=(np.abs(x)<=norm.ppf(.975)*math.sqrt(vf)).mean(0)
    np.testing.assert_allclose(covp,expected['persistent_coverage_under_augmented95'],atol=2/len(x))
    np.testing.assert_allclose(covn,expected['persistent_coverage_under_native95'],atol=2/len(x))
 for row in cost['rows']:
  for field in ['native_factorized','ordinary_dense_augmented']:
   assert len(row[field]['timings'])==3
   assert all(v>0 and np.isfinite(v) for t in row[field]['timings'] for v in t.values())
   assert row[field]['peak_increment_bytes']>=0
 result={'all_arithmetic_checks_passed':True,'native_rows':checks,'negative_control_or_one_step_rows':negative,
         'max_native_vs_fresh_variance_error':max_native,'max_joint_vs_persistent_variance_error':max_joint,
         'simultaneous_MC_variance_intervals':mc,'individual_interval_alpha':alpha,
         'expected_variance_inside_intervals_count':sum(sum(x['expected_inside_CI']) for x in mc),'total_variance_intervals':len(mc)*2,
         'interval_scope':'Bonferroni familywise .05 across168 coordinate/family/horizon sample-variance intervals; descriptive MC verification, no tuning',
         'cost_rows_checked':len(cost['rows']),'cost_scope':'Current eager inference operators; no equal-optimization or training-cost claim',
         'operator_results_sha256':sha(OUT/'operator/results.json'),'cost_results_sha256':sha(OUT/'cost/results.json'),
         'trained_model_relevance_established':False,'objective_achieved':False,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 out=OUT/'independent_verification.json';assert not out.exists();out.write_text(json.dumps(result,indent=2),encoding='utf-8')
 print(json.dumps({k:v for k,v in result.items() if k!='simultaneous_MC_variance_intervals'}))
if __name__=='__main__':main()
