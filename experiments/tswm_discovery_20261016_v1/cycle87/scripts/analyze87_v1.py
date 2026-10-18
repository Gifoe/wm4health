from common87_v1 import *
import numpy as np
FAMILIES=['duffing','stuart_landau'];SEEDS=[2026872,2026873,2026874]
def main():
 result=read(O/'EVALUATION_RESULTS87_V3.json');folder=RAW/'evaluation_v2';rng=np.random.default_rng(20268799);records=[];interactions=[];summaries=[]
 for fam in FAMILIES:
  fi=FAMILIES.index(fam);by_cohort={}
  for cohort in ['retained','crossed','unselected']:
   with np.load(folder/f'{fam}_{cohort}_v1.npz') as z:truth=z['truth'][:,1:]
   denominator=np.maximum(np.sqrt(np.mean(truth.astype(float)**2,axis=(1,2))),.1);errors={}
   for arm in ['whole_filter','resampled_ic','unselected']:
    values=[];raw=[]
    for seed in SEEDS:
     row=next(r for r in result['rows'] if r['family']==fam and r['cohort']==cohort and r['arm']==arm and r['seed']==seed)
     with np.load(row['array_path']) as z:pred=z['prediction'][fi*128:(fi+1)*128,1:].astype(float)
     assert np.isfinite(pred).all(),'Native failure requires explicit alternative failure-aware analysis'
     mse=np.mean((pred-truth)**2,axis=(1,2));raw.append(mse);values.append(np.sqrt(mse)/denominator)
    errors[arm]=dict(nrmse=np.array(values),mse=np.array(raw));summaries.append(dict(family=fam,cohort=cohort,arm=arm,mean_nrmse=float(np.mean(values)),optimizer_seed_nrmse_means=np.mean(values,axis=1).tolist(),mean_mse=float(np.mean(raw))))
   draw=rng.integers(0,128,(2000,128));delta=np.mean(errors['resampled_ic']['nrmse']-errors['whole_filter']['nrmse'],axis=0);raw_delta=np.mean(errors['resampled_ic']['mse']-errors['whole_filter']['mse'],axis=0);boot=delta[draw].mean(axis=1);raw_boot=raw_delta[draw].mean(axis=1)
   records.append(dict(family=fam,cohort=cohort,resampled_minus_filtered_mean_nrmse=float(delta.mean()),exploratory_conditional_pool_bootstrap_interval=np.percentile(boot,[2.5,97.5]).tolist(),optimizer_seed_delta_mean_nrmse=np.mean(errors['resampled_ic']['nrmse']-errors['whole_filter']['nrmse'],axis=1).tolist(),resampled_minus_filtered_mse=float(raw_delta.mean()),exploratory_mse_interval=np.percentile(raw_boot,[2.5,97.5]).tolist()))
   by_cohort[cohort]=delta
  interaction=by_cohort['crossed']-by_cohort['retained'];draw=rng.integers(0,128,(2000,128));interactions.append(dict(family=fam,crossed_minus_retained_resampling_effect=float(interaction.mean()),exploratory_conditional_pool_interval=np.percentile(interaction[draw].mean(axis=1),[2.5,97.5]).tolist()))
 wins=0;total=0
 for row in result['rows']:
  if row['arm'] not in ['whole_filter','resampled_ic','unselected']:continue
  name='empirical_prior_unselected' if row['arm']=='unselected' else 'empirical_prior_whole_filter';control=next(r for r in result['rows'] if r['model']==name and r['family']==row['family'] and r['cohort']==row['cohort']);wins+=control['horizons'][-1]['failure_penalized_mean_nrmse']<row['horizons'][-1]['failure_penalized_mean_nrmse'];total+=1
 report=dict(utc=utc(),status='POSTSTUDY_DESCRIPTIVE_ANALYSIS_NOT_MODEL_SELECTION',source_results_sha256=sha(O/'EVALUATION_RESULTS87_V3.json'),seed_averaged_pairs=records,interactions=interactions,native_summaries=summaries,matched_empirical_prior_wins=wins,total_matched_native_comparisons=total,bootstrap_scope='Exploratory conditional on one fixed TRAIN pool and one held-out pool per family; averages3 optimizer seeds; no source-population uncertainty or confirmatory p-values',script_sha256=sha(__file__))
 write(O/'PAIRED_EFFECT_ANALYSIS87_V1.json',report)
 lines=['# Closed paired effects and ordinary-control decision','',f'Matched-source empirical-prior H10 failure-aware NRMSE is lower in {wins}/{total} native comparisons. This is descriptive within the two bounded cubic families, not a universal statement about all world models.','', '| family | cohort | B-minus-A mean NRMSE | exploratory conditional-pool interval |3 optimizer-seed differences |','|---|---|---:|---|---|']
 for r in records:lines.append(f"| {r['family']} | {r['cohort']} | {r['resampled_minus_filtered_mean_nrmse']:.6g} | {r['exploratory_conditional_pool_bootstrap_interval']} | {r['optimizer_seed_delta_mean_nrmse']} |")
 lines+=['','B is resampled_ic, A is whole_filter; negative favors resampling. Intervals condition on the fixed TRAIN pool, average the3 optimizer seeds and bootstrap128 held-out mechanisms; they are exploratory, not confirmatory population-level tests. No hyperparameter changed after test opening.','', '## Crossed-versus-retained interaction','']
 for r in interactions:lines.append(f"- {r['family']}: {r['crossed_minus_retained_resampling_effect']:.6g}; exploratory interval {r['exploratory_conditional_pool_interval']}.")
 lines+=['','The crossed test preserves mechanism/IC marginals but changes state coverage/information, so an interaction alone is not a shortcut proof. High source acceptance and ordinary prior-aware sufficiency limit practical and methodological claims. Selection-aware amortized inference already has close prior art (NOVELTY_ADDENDUM87_V1.md). The scoped new-method route is not supported. This does not falsify all possible effects of trajectory filtering or erase the C86 linear construction.','', 'No new architecture or final research direction is accepted. Any later direction must start from a concrete failure left unresolved by competent ordinary and prior-aware controls.']
 with (C/'POSTSTUDY_ANALYSIS87_V1.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
 log(dict(stage='paired_descriptive_analysis',matched_empirical_wins=wins,total=total,rows=records,interactions=interactions))
if __name__=='__main__':main()
