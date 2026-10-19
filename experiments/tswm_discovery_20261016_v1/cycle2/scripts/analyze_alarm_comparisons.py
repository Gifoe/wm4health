"""Prespecified comparisons and paired uncertainty; no test-based selection."""
import hashlib,json,pathlib
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs'
def main():
 seeds=[41,42,43];records=[];rng=np.random.default_rng(20261016);sources={}
 for kind in ['state_dependent_alarm','independent_alarm_control']:
  with np.load(OUT/'native_alarm'/(kind+'_data.npz')) as f:mask=f['test_mask'];alarm=f['test_alarm']
  native=[np.load(OUT/'native_alarm_extended'/(kind+f'_seed{s}_evaluation_arrays.npz')) for s in seeds]
  generative=[np.load(OUT/'generative_alarm'/(kind+f'_seed{s}_evaluation_arrays.npz')) for s in seeds]
  rseeds=seeds if kind=='state_dependent_alarm' else [41]
  positive=[np.load(OUT/'rpm_exact'/(kind+f'_positive_site_seed{s}_evaluation_arrays.npz')) for s in rseeds]
  full=[np.load(OUT/'rpm_exact'/(kind+f'_full_gaussian_seed{s}_evaluation_arrays.npz')) for s in rseeds]
  for L,H in [(16,1),(1,0)]:
   selected=alarm[:,L-1]&~mask[:,L-1];idx=np.where(selected)[0]
   for other,label,other_files,paired_seeds in [('exact_same_readout','retained_F',native,seeds),('generative_ssm','ordinary_generative_SSM',generative,seeds),('positive_site_exact_ML','ordinary_RPM_positive_exact_ML',positive,rseeds),('full_gaussian_exact_ML','ordinary_RPM_full_exact_ML',full,rseeds)]:
    for metric in ['mse','nll','coverage','tail_brier']:
     key=f'L{L}_H{H}_';baseline=np.stack([f[key+'native_'+metric][idx] for f in native[:len(paired_seeds)]])
     alternate=np.stack([f[key+other+'_'+metric][idx] for f in other_files]);delta=alternate-baseline
     draws=[]
     for _ in range(3000):
      iseed=rng.integers(0,len(paired_seeds),size=len(paired_seeds));jseed=iseed if other=='exact_same_readout' else rng.integers(0,len(paired_seeds),size=len(paired_seeds));icase=rng.integers(0,len(idx),size=len(idx))
      draws.append(float(alternate[jseed][:,icase].mean()-baseline[iseed][:,icase].mean()))
     records.append({'dataset':kind,'prefix':L,'horizon':H,'subset':'alarm_no_anchor','comparison':label+' minus native_10000','metric':metric,
      'n_test_episodes':len(idx),'initialization_seeds':paired_seeds,'mean_difference':float(delta.mean()),'per_seed_difference':delta.mean(axis=1).tolist(),
      'resampling':'paired test episodes; paired initializations for same frozen checkpoint' if other=='exact_same_readout' else 'paired test episodes; independent initialization resampling across separately trained methods',
      'paired_hierarchical_bootstrap_95_interval':np.percentile(draws,[2.5,97.5]).tolist(),'bootstrap_replicates':3000})
  for file in native+generative+positive+full:file.close()
 for folder in ['native_alarm_extended','generative_alarm','rpm_exact']:
  for file in sorted((OUT/folder).glob('*_evaluation.json')):sources[file.relative_to(ROOT).as_posix()]=hashlib.sha256(file.read_bytes()).hexdigest()
 result={'primary':'prefix16,horizon1,alarm_no_anchor; prefix1/horizon0 is a diagnostic, not a replacement primary',
  'uncertainty_scope':'Paired test episodes on one fixed training dataset. Initializations are paired only for the same frozen checkpoint; separately trained methods resample initializations independently. RPM independent-alarm controls only use seed41. Not independent-data replication, no simultaneous multiplicity guarantee, no test-selected method.',
  'records':records,'input_sha256':sources,'code_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}
 (OUT/'alarm_paired_comparisons.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps([r for r in records if r['prefix']==16 and r['dataset']=='state_dependent_alarm' and r['metric'] in ['nll','tail_brier']],indent=2))
if __name__=='__main__':main()
