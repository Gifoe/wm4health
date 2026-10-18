"""Independent score arithmetic, paired record uncertainty, and bounded conclusions."""
from common61 import *
import numpy as np,csv,io

def interval(v,indices):
 means=v[indices].mean(-1);return [float(x) for x in np.quantile(means,[.025,.975])]

def main():
 sources();e=read(OUT/'EVALUATION61_V1.json');seal=read(OUT/'MODEL_SEAL61_V1.json');opened=read(OUT/'TEST_OPEN61_V1.json');data=read(OUT/'DATA61_V1.json');q=read(OUT/'QUALIFICATION61_V2.json')
 assert e['status']=='SEALED_EXISTING_HSMM_AND_HMM_SCORED_WITH_INDEPENDENT_REPLAY'
 dt=datetime.datetime.fromisoformat;assert dt(seal['utc'])<dt(opened['utc'])<dt(e['utc']);assert e['seal_sha256']==opened['seal_sha256']==sha(OUT/'MODEL_SEAL61_V1.json');assert len(seal['models'])+len(seal['failed'])==12
 oracle={};loaded={};training=[];err=0.
 for r in e['oracle']:
  assert sha(r['arrays'])==r['arrays_sha256']
  with np.load(r['arrays']) as a:oracle[r['family']]={k:a[k].copy() for k in a.files}
 for r in e['records']:
  assert sha(r['arrays'])==r['arrays_sha256'];recpath=OUT/'runs'/r['tag']/'result.json';rec=read(recpath);saved=next(x for x in seal['models'] if x['tag']==r['tag']);assert sha(recpath)==saved['result_sha256'];assert sha(rec['selected']['checkpoint'])==r['checkpoint_sha256'];assert sha(rec['terminal_checkpoint'])==rec['terminal_checkpoint_sha256'];assert sha(OUT/'runs'/r['tag']/'curve.jsonl')==rec['curve_sha256'];assert rec['updates_completed']==1500 and not rec['audit_truth_read'] and not rec['test_read'] and not rec['test_score_selection'];training.append(rec)
  with np.load(r['arrays']) as a:values={k:a[k].copy() for k in a.files}
  o=oracle[r['family']];assert np.array_equal(values['end_event'],o['end_event']);assert np.array_equal(values['oracle_survival_probability'],o['survival_probability'])
  b=(1-values['survival_probability']-values['end_event'])**2;p=(values['survival_probability']-o['survival_probability'])**2;delta=values['conditional_nll']-o['conditional_nll']
  metrics={'conditional_nll':values['conditional_nll'].mean(),'conditional_nll_above_oracle':delta.mean(),'brier':b.mean(),'survival_probability_mse_vs_oracle':p.mean(),'observation_mse':values['observation_mse'].mean()}
  for k,v in metrics.items():err=max(err,abs(float(v)-r[k]));assert abs(float(v)-r[k])<1e-12,(r['tag'],k)
  loaded[r['tag']]={**values,'delta_nll_record':delta.mean(1),'brier_record':b.mean((1,2)),'probability_mse_record':p.mean((1,2))}
 rng=np.random.default_rng(63100);indices=rng.integers(0,128,size=(5000,128));groups={};pairs={}
 for family in ['peaked','bimodal']:
  groups[family]={}
  for arm in ['hsmm','hmm']:
   rows=[r for r in e['records'] if r['family']==family and r['arm']==arm];complete=len(rows)==3 and all(r['finite'] for r in rows);g={'complete_three_seed_group':complete,'fits':len(rows),'finite_fits':sum(r['finite'] for r in rows),'per_seed':[{k:r[k] for k in ['seed','selected_step','conditional_nll','conditional_nll_above_oracle','brier','survival_probability_mse_vs_oracle','observation_mse']} for r in rows]}
   for k in ['conditional_nll','conditional_nll_above_oracle','brier','survival_probability_mse_vs_oracle','observation_mse']:
    g[k+'_mean']=float(np.mean([r[k] for r in rows])) if complete else None;g[k+'_seed_sd']=float(np.std([r[k] for r in rows],ddof=1)) if complete else None
   if complete:
    nd=np.mean([loaded[r['tag']]['delta_nll_record'] for r in rows],0);g['nll_above_oracle_record_bootstrap95']=interval(nd,indices)
   if arm=='hsmm':g['all_seeds_pass_prespecified_sufficiency_screen']=bool(complete and all(r['conditional_nll_above_oracle']<=.05 and r['survival_probability_mse_vs_oracle']<.01 for r in rows))
   groups[family][arm]=g
  if all(groups[family][a]['complete_three_seed_group'] for a in ['hsmm','hmm']):
   rr={a:[r for r in e['records'] if r['family']==family and r['arm']==a] for a in ['hsmm','hmm']};d=np.mean([loaded[r['tag']]['conditional_nll'].mean(1) for r in rr['hsmm']],0)-np.mean([loaded[r['tag']]['conditional_nll'].mean(1) for r in rr['hmm']],0);pairs[family]={'hsmm_minus_hmm_conditional_nll':float(d.mean()),'paired_record_bootstrap95':interval(d,indices),'conditioning':'All three fitted seeds fixed; records resampled with all history/horizon views together. This is not uncertainty over training seeds.'}
 sufficient=all(groups[f]['hsmm']['all_seeds_pass_prespecified_sufficiency_screen'] for f in groups)
 decision='EXISTING_LEARNED_HSMM_SUFFICIENT_ON_BOTH_CONTROLLED_FAMILIES' if sufficient else 'SUFFICIENCY_SCREEN_NOT_PASSED_ATTRIBUTION_REQUIRED_NO_NEW_GAP_ESTABLISHED'
 summary={'utc':utc(),'status':'C61_EXISTING_DURATION_CONTROL_INTERPRETED','decision':decision,'groups':groups,'paired_arm_comparisons':pairs,'failed_attempts':seal['failed'],'valid_fits':len(training),'valid_neural_optimizer_updates':sum(r['updates_completed'] for r in training),'qualification_updates_separate':q['profile_updates_separate'],'max_score_arithmetic_discrepancy':err,'max_native_dense_conditional_nll_error':max(r['max_native_vs_independent_conditional_nll_error'] for r in e['records']),'max_native_dense_posterior_error':max(r['max_native_vs_independent_posterior_error'] for r in e['records']),'max_native_dense_survival_error':max(r['max_native_vs_independent_survival_error'] for r in e['records']),'evaluation_sha256':sha(OUT/'EVALUATION61_V1.json'),'real_domains':0,'overall_goal':'ACTIVE_UNACHIEVED','accepted_final_directions':0}
 dump(OUT/'SUMMARY61_V1.json',summary)
 buf=io.StringIO(newline='');fields=['family','arm','seed','selected_step','parameters','trainable_parameters','updates','training_seconds','conditional_nll','conditional_nll_above_oracle','brier','survival_probability_mse_vs_oracle','observation_mse'];w=csv.DictWriter(buf,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(e['records'])
 with (OUT/'results61_v1.csv').open('x',encoding='utf-8',newline='') as f:f.write(buf.getvalue())
 table=['| Family | Existing model | Fits | Conditional joint NLL/step | NLL above known law | Remaining-life probability MSE | Brier |','|---|---|---:|---:|---:|---:|---:|']
 for family in groups:
  for arm,g in groups[family].items():
   values=['N/A' if g[k+'_mean'] is None else f"{g[k+'_mean']:.7g}" for k in ['conditional_nll','conditional_nll_above_oracle','survival_probability_mse_vs_oracle','brier']];table.append(f"|{family}|{arm}|{g['fits']}/3|"+'|'.join(values)+'|')
 table='\n'.join(table)
 outcome=('The learned existing HSMM passes the predeclared narrow sufficiency screen in every fitted seed on both families. Generic residual-life belief is not supported as a new-method necessity here.' if sufficient else 'At least one predeclared sufficiency condition fails. This does not establish a research gap; inspect parameter learning, finite data, structural assumptions and optimization before proposing a mechanism.')
 boundary='Two synthetic finite-duration Gaussian-emission families, fixed known state count/support, fresh known episode boundaries, independent records and unknown fitted law parameters. No real domain, unknown observation model, unknown regime count, arbitrary left censoring, cross-system pretrained checkpoint or new architecture is evaluated. The term physical_test_records in the machine evaluation denotes independent generated paths; it does not mean real measurements.'
 reports={
 'README.md':f'# C61: learned existing duration controls\n\n{outcome}\n\n{table}\n\nEvery seed and source/hash/config is retained. See outputs/results61_v1.csv, SUMMARY61_V1.json and EVALUATION61_V1.json. Overall goal remains active and unachieved; accepted final directions = 0.\n\n{boundary}\n',
 'literature_review.md':'''# C61 prior-art boundary

[NHSMM source](https://github.com/awa-si/nhsmm/tree/25aeaaff594c0250359226145d7192f4ec8c427f) already implements joint state-age causal filtering, context modulation, native likelihood learning, streaming state, repeated survival queries and active-episode remaining-life probabilities. These are direct existing mechanisms. The current native component execution does not reproduce an associated paper's table or the package's full optimizer/default context encoder. The fixed budget optimizer and supported injected transition subclass are explicit adapters.

[Linderman SSM](https://github.com/lindermanlab/ssm/tree/eb6c8aa33e5311d3564075807dec340759dd8081) and [Bayesian Nonparametric Hidden Semi-Markov Models](https://arxiv.org/abs/1203.1365) provide older explicit-duration and state-space precedents. Linderman source is audited and pinned but not executed in this cycle. No claim of reproducing its Cython/Autograd runtime is made. NHSMM Apache-2.0 and SSM MIT licenses are retained with original bytes. A learned semi-Markov posterior cannot be renamed into a contribution merely by calling it a world state.

Native active-episode survival freezes the duration distribution at the current information set; it does not synthesize unknown future context. This cycle uses autonomous stationary laws, so that contract is appropriate. Future action-dependent hazards and observation models require separately specified conditional laws and matched existing controls; lack of such experiments is not proof of a new gap.
''',
 'baseline_registry.md':f'''# C61 executed identities

Unmodified NHSMM causal likelihood and filtering/survival code, pinned to the commit in source_selection61_v1.json. HSMM: K=3, D=32, unknown categorical duration/transition/diagonal Gaussian emission parameters, no internal context encoder. A supported injected Transition subclass forbids unobservable same-state renewal. HMM: same source with D=1 and unrestricted transitions; same-state transition means persistence for its remaining-life endpoint. All fitted parameters use observation-only likelihood; both arms get native train-only k-means emission initialization and the same record sampler seed.

Twelve fixed attempts, 1500 Adam updates each, three CPU workers with four threads per worker, validation every 100 including zero, full individual-seed outcomes. Synthetic CPU profiling median {q['synthetic_step_seconds']['cpu']:.6g}s versus CUDA {q['synthetic_step_seconds']['cuda']:.6g}s chose CPU before data outcomes. Qualification optimizer steps are separate. Parameter counts and runtime differ across arms; this is a competent existing-model exclusion, not equal-capacity or equal-compute novelty evidence.

Direct import used existing Python {q['python']} below package metadata >=3.12. Qualified paths executed successfully without package/environment changes; no supported installation or full-package compatibility claim. Failed preflight V1 changed a private ergodic support mask and conflicted with a runtime fast path. V2 uses the documented distribution-factory extension. Preserve both attempts; the V1 discrepancy is our adapter issue, not native model insufficiency.
''',
 'experimental_results.md':f'''# C61 complete evidence

{table}

{outcome}

Valid fixed-budget optimizer updates: {summary['valid_neural_optimizer_updates']}; separate profiling updates: {summary['qualification_updates_separate']}. Known-law reference parameters and latent labels are never provided to training or validation selection. Primary joint future observation NLL and secondary remaining-life probability/Brier arithmetic are independently verified from saved arrays. Native versus independent probability-space recursion maximum conditional-NLL discrepancy: {summary['max_native_dense_conditional_nll_error']:.8g}; posterior discrepancy: {summary['max_native_dense_posterior_error']:.8g}; survival discrepancy: {summary['max_native_dense_survival_error']:.8g}. Posterior inference receives only its observed prefix. Future observations enter a separate proper density score.

Bootstrap intervals resample 128 independent generated paths while keeping each path's three history lengths and five survival horizons together, conditioning on the three fixed model seeds. Seed SD is reported separately in JSON. Model seal precedes test opening and no test values choose a new fit or recipe. No model instability or missing fit is silently averaged away.

{boundary}
''',
 'negative_evidence.md':f'''# C61 negative evidence

{outcome}

An HMM-versus-HSMM difference on an HSMM-generated family is an established duration-modeling effect. It supplies no novel world-model mechanism. Known-law filtering is a numerical/reference ceiling with privileged law parameters; it is not evidence that unknown-parameter learning is effortless. Actual unlabeled learning is reported separately for all seeds.

The correct-support likelihood, fixed K and finite duration range are substantial structural priors. Observation-space likelihood does not automatically certify arbitrary latent semantics. Remaining-life truth is defined by this controlled generator; unknown real systems may not identify the same episodes. No nonlinear observation model, informative sensing, action-dependent renewal, left-censored start or real-data device population is tested. These omissions are limits, not declarations that existing approaches fail there.

The failed first adapter, unsupported Python metadata version and source-component versus full-package training distinction remain visible. None is a publishable mechanism failure. Overall research goal remains active and unachieved.
''',
 'hypothesis_space.md':f'''# C61 competing explanations

1. Generic regime-duration uncertainty requires a new neural world-model mechanism: unsupported by prior art; {outcome}
2. Ordinary explicit-duration inference can learn useful remaining-life posteriors without age/regime labels: directly tested by six HSMM fits with observation-only train/validation likelihood.
3. A poor memoryless latent baseline demonstrates novelty: rejected. HMM duration restrictions are known, and the existing HSMM is the necessary control.
4. Any poor residual forecast proves structural insufficiency: rejected. Unknown parameter estimation, sample size, segmentation equivalence, initialization and finite optimization remain competing explanations.
5. A reusable new framework has been validated: false. These are separately fitted existing components on controlled families; accepted directions remain zero.
''',
 'research_ideas.md':f'''# C61 direction disposition

{outcome}

Do not add an age head, duration hazard, semi-Markov latent or state-age belief and call that a new framework. A useful future contribution would require a distinct residual or efficiency advantage beyond learned explicit-duration models, competent RSSM/joint-law controls and fair observation likelihoods, with independent real evidence. That evidence is not supplied here. Failure of this sufficiency screen would require attribution, not immediate architecture invention.

C60 already found that ordinary neural state-space models explain the measured mixed-memory deficit. C61 checks the other narrow C58 exclusion question. These local exclusions do not prove the broad field solved, and they do not lower the original goal to reproducing known controls. Accepted final directions remain zero.
''',
 'final_research_direction.md':f'''# C61 final research-direction status

**Research Question:** Is posterior remaining-life uncertainty a demonstrated missing capability of existing time-series world-model building blocks?

**Evidence:** Native likelihood/filter/survival and gradients qualified against independent probabilities, supported streaming state agreement, twelve sealed fixed-budget unlabeled fits, proper joint future observation density, conditional survival and all-seed/independent-path audit. {outcome}

{table}

**Existing Methods:** Explicit-duration HSMMs already expose the proposed state-age belief and residual-life mechanism. The current public NHSMM source executes that mechanism; older SSM/HDP-HSMM precedents reinforce the boundary.

**Proposed Idea:** None accepted. This is an existing-control qualification and hypothesis exclusion, not a new framework.

**Falsification:** Where learned existing controls meet the declared capability, a claimed new duration mechanism is unnecessary. Where they miss, optimization and model-family assumptions must be distinguished before calling it a gap.

**Feasibility:** Existing CPU/Torch runtime, small source-controlled models, all data/checkpoints/predictions on authorized G storage, compact code/evidence on GitHub; no package upgrades.

**Next Experiment:** Return to failure discovery on independent source-grounded measured systems or validate a concretely different capability with competent existing controls. Do not retune these opened synthetic test splits or claim all nonstationary/controlled duration problems have been solved.

{boundary}

Overall goal ACTIVE_UNACHIEVED. Accepted final directions: 0.
'''}
 for name,text in reports.items():
  with (ROOT/name).open('x',encoding='utf-8') as f:f.write(text)
 closure={'utc':utc(),'status':'C61_LEARNED_EXISTING_DURATION_CONTROL_SCREEN_CLOSED','decision':decision,'summary_sha256':sha(OUT/'SUMMARY61_V1.json'),'evaluation_sha256':sha(OUT/'EVALUATION61_V1.json'),'reports_sha256':{k:sha(ROOT/k) for k in reports},'source_objects_verified':25,'selected_and_terminal_checkpoint_hashes_verified':True,'real_domains':0,'overall_goal':'ACTIVE_UNACHIEVED','accepted_final_directions':0}
 dump(OUT/'CLOSURE61_V1.json',closure);print(json.dumps(closure),flush=True)

if __name__=='__main__':main()
