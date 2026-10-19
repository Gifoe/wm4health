"""Preserve a native failure and its ordinary-control falsification without novelty inflation."""
import runtime_profile
import json,pathlib,statistics,datetime
ROOT=pathlib.Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def put(name,value):
 p=ROOT/name;assert not p.exists();p.write_text(value.strip()+'\n',encoding='utf-8')
def main():
 r=read(ROOT/'outputs/operator/results.json');v=read(ROOT/'outputs/independent_verification.json');cost=read(ROOT/'outputs/cost/results.json')
 assert v['all_arithmetic_checks_passed'] and v['native_rows']==630
 table='| a | h | Native variance | Persistent variance | Actual coverage of native nominal95% interval |\n|---:|---:|---:|---:|---:|\n'
 for a in [1.,.95,-.8,0.]:
  x=next(x for x in r['rows'] if x['native_branch']=='worker' and x['a']==a and x['c']==1 and x['context_variance']==.5 and x['horizon']==64)
  table+=f"| {a:g} | 64 | {x['native_variance']:.6f} | {x['persistent_variance']:.6f} | {100*x['persistent_nominal95_actual_marginal_coverage']:.3f}% |\n"
 times='| Latent observation dim | Batch | Native eager ms/32 steps | Dense augmented ms/32 steps | Augmented peak increment MiB |\n|---:|---:|---:|---:|---:|\n'
 for x in cost['rows']:
  n=statistics.median(z['wall_ms_per_rollout'] for z in x['native_factorized']['timings']);a=statistics.median(z['wall_ms_per_rollout'] for z in x['ordinary_dense_augmented']['timings'])
  times+=f"| {x['latent_obs_dim']} | {x['batch']} | {n:.3f} | {a:.3f} | {x['ordinary_dense_augmented']['peak_increment_bytes']/1048576:.3f} |\n"
 put('literature_review.md','''# Persistent context: primary-source and innovation boundary

This is a general probabilistic time-series world-model question. A dataset-specific performance correction is not the objective. A point-forecast model is not required to provide a stochastic simulation interface.

- [Multi Time Scale World Models](https://arxiv.org/pdf/2310.18534), NeurIPS2023: fast dynamics use one slow context within a window. Eq5 gives an additive linear-Gaussian conditional transition; Appendix Eq14 propagates state and context marginal covariances separately. Official [source](https://github.com/ALRhub/MTS3) is pinned at `ab971c0ec43df7664c4b727ca69d5b8dd0cab5da`. This cycle measures an approximation of that moment recurrence, not a divergence between paper and implementation. Within-fast-state covariance factorization is deliberately exact in the diagnostic dimension, isolating omitted cross-scale dependence.
- [HiP-RSSM](https://arxiv.org/pdf/2206.14697v3), ICLR2022: a Gaussian latent task/context conditions state filtering. The official original source at `f4adef0ce5ed4b37c9847973df320f72965bd330` was inspected in a selected16-file snapshot. Its contextual prediction supports learned mean/variance transformations. The native numeric tests here use the newer MTS3 repository's worker and HIPRSSM branches; those are two branches of one shared source implementation, not two independently reproduced trained models.
- [PETS](https://proceedings.neurips.cc/paper/7725-deep-reinforcement-learning-in-a-handful-of-trials-using-probabilistic-dynamics-models.pdf), NeurIPS2018, Section5.1: TS-infinity retains a dynamics-model identity over the whole trajectory. Persistent model sampling and separating epistemic from per-transition noise are established ideas. A proposed "sample a mechanism once" method would overlap this directly.
- [Explore the Context](https://proceedings.mlr.press/v130/achterhold21a.html), AISTATS2021: stochastic context-conditional dynamics and information-gain-based calibration already connect latent mechanism posteriors to system exploration. A probabilistic context encoder is not an invention.
- Ordinary augmented-state linear-Gaussian inference retains state/context cross-covariance. We implement this standard mathematical control explicitly, rather than call it a new world-model architecture. The test is elementary joint Gaussian propagation with no observations; it does not introduce a new filtering theorem.

All four primary PDFs are archived under outputs/literature with SHA256. The MTS3 versionless arXiv URL is content-pinned without guessing its revision. This review does not prove the absence of more sophisticated recent work. Previous cycle14 already reviewed shared-mechanism OneWorld, stochastic JEPA and Bayesian neural-process overlap; those claims remain separate from this native operator test.
''')
 put('baseline_registry.md','''# Actual execution and provenance

## Executed native source

Official ALRhub/MTS3 revision `ab971c0ec43df7664c4b727ca69d5b8dd0cab5da`, 152 source/config/document files, no AGENTS files or detected hardcoded credentials. Import only the inspected Gaussian marginalization module. No native experiment launcher, W&B logging, downloaded pickle or pretrained checkpoint was executed.

`Predict` worker and HIPRSSM are exercised with prescribed admissible parameters. Both use the same source file and covariance formula. latent_obs_dim1 means a two-dimensional fast state and two-dimensional context; block covariance within each is exact in this experiment. This is a native implementation discriminator, not a trained model or original-paper benchmark reproduction.

## Strong ordinary controls actually executed

1. Closed-form marginal moments of a single persistent Gaussian context.
2. Closed-form marginal moments of independently redrawn contexts.
3. Standard dense augmented-state Gaussian propagation retaining fast/context cross-covariance.
4. Independent NumPy simulation, 50,000 particles per preselected case, and independent arithmetic/interval verification from saved arrays.
5. CUDA eager inference cost for native factorized vs conventional augmented propagation,6 preselected dimension/batch configurations.

## Source inspection only

Selected16-file original HiP-RSSM source at `f4adef0ce5ed4b37c9847973df320f72965bd330`. This is not a complete original checkout and none of it was executed. Other models reviewed in earlier cycles have not been retrained here.

## Dataset and training status

No dataset-based fit was run in cycle15. The publisher MTS3 processed-data Google Drive URL timed out from both hosts. The original HiP-RSSM numeric source was located and its Git blob pinned, but an optional local download did not complete; the later live request was deliberately stopped to prioritize the ordinary-control cost gate. No data was deserialized or substituted and no original-paper data reproduction is claimed. These acquisition issues did not block the native operator experiment.
''')
 put('experimental_results.md',f'''# Persistent-context marginalization discriminator

## Controlled model

For fast state z and persistent context l, z_next = a*z + c*l + epsilon. Start with independent zero-mean state variance.05 and context variance s; process variance.01. Context is sampled once for a rollout. The source-native approximation adds c^2*s every step without carrying fast/context cross-covariance.

At horizon h, exact persistent marginal variance is a^(2h)*P0 + c^2*(sum_j a^j)^2*s + q*sum_j a^(2j). Independently resampling the context gives a^(2h)*P0 + (c^2*s+q)*sum_j a^(2j). Standard augmented-state Gaussian propagation reproduces the former. All means agree in this zero-mean example; only uncertainty differs.

## Native evidence

90 prescribed cell/branch rollouts,7 inspected horizons each:630 rows. Native and fresh-context marginal formulas agree to {v['max_native_vs_fresh_variance_error']:.3g}; augmented and persistent formulas to {v['max_joint_vs_persistent_variance_error']:.3g}. There are438 one-step or negative-control rows. Negative a tests cancellation and can yield overdispersion; failure is not universally undercoverage.

{table}
These are analytic Gaussian marginal coverages for prescribed native operator parameters, not percentages from trained models on a dataset. a=0 is only a marginal-variance negative control; temporal dependence may still differ across processes. No unique full joint law is inferred just from matching marginal moments.

Independent simulations:6 preselected cases x50,000 particles,42 horizon rows. All168 simultaneous coordinate/family/horizon sample-variance confidence intervals contain the known variance, using Bonferroni familywise.05. Saved endpoints independently reproduce variances and coverage statistics. This is verification, not test-set tuning.

## Conventional-control cost

RTX5090, float32/eager, TF32 disabled,32 open-loop steps. Warm-up4, three repeats of5 rollouts. CUDA event times and synchronized wall times are both saved.

{times}
At the native d30 configuration, full augmented dimension is120. The ordinary joint method is practical in this inference test. Relative timings compare current eager code paths with different retained information, not equally optimized algorithms. The native path recomputes several constant transforms and uses more small operations; ordinary hoisting, compilation or kernel fusion could change its relative time. These results do not establish a general algorithmic speedup. Peak allocated memory increments are inference-only, not training/backpropagation cost.

## What remains unproved

No trained native model, learned context-channel relevance, real-data interval coverage, decoded joint-path calibration, training memory/throughput, arbitrary nonlinear mechanism or cross-dataset improvement has been evaluated. A learned model may reduce context coupling or encode persistent information in its fast state. The operator witness alone cannot establish an important failure of the complete learned model.
''')
 put('negative_evidence.md','''# Negative evidence and candidate decisions

1. The native recurrence misses persistent state/context dependence in an admissible controlled Gaussian case. This is a reproducible approximation limitation, not yet an unsolved research gap.
2. A conventional augmented state gives the exact answer. Its eager inference cost and memory are already practical at the canonical native dimension. The hypothesis that this case requires a new compressed covariance architecture is unsupported.
3. PETS TS-infinity already preserves a model identity along a trajectory. A persistent global latent or fixed ensemble index is not new.
4. Large analytic undercoverage in a prescribed cell does not show trained-model undercoverage. Learned representations may carry the relevant variable inside the fast state or suppress the context channel. Do not infer learned failure from the cell witness.
5. The two native branches share one implementation;630 rows are not630 independent datasets or90 trained models. Monte Carlo confidence checks verify a known mathematical process and do not establish statistical gains of a learned method.
6. Dense joint inference timings do not establish training feasibility at the publisher's large batch sizes or justify a claim of optimized algorithmic superiority. No matched low-level optimization study was conducted.
7. Original data acquisition did not complete; no processed-data equivalence, data result or full reproduction is implied. No important earlier result, checkpoint or environment was deleted.

Decision: do not develop or rename an ordinary joint Gaussian filter as a new architecture on this evidence. Preserve the native finding and the strong ordinary control. Accepted research directions remain zero.
''')
 put('hypothesis_space.md','''# Hypotheses and falsification

| Hypothesis | Test | Evidence/decision |
|---|---|---|
| Native repeated marginal propagation preserves the persistent-context marginal law | Native cell vs exact joint Gaussian and closed form | Rejected in192 non-control/horizon rows; remaining438 one-step/negative-control rows agree |
| The difference is a spatial covariance approximation within fast state | Use latent_obs_dim1 with exactly representable2x2 covariance | Excluded for this witness |
| The native marginal formula instead agrees with fresh-context moments | Independent fresh-context formula and MC | Supported; this does not uniquely identify an entire path law |
| A new expensive or compressed filter is necessary at the native dimension | Conventional dense augmented control on one5090 | Unsupported: canonical inference cost is low |
| Trained native models have consequential decoded miscalibration from this term | Faithful native training, inspect learned C/context variance, compare controlled inference with held-out forecast calibration | Untested; no learned-model failure claim |
| Existing posterior-reuse methods remain insufficient under realistic partial observation at matched finite cost | Existing Bayesian/particle/latent controls with an identifiable data contract | Still unresolved; the broad question is not an accepted innovation |

The next scientific gate, if this audit is pursued further, is trained-model relevance and whether the ordinary correction suffices. If ordinary joint or persistent-sampling inference solves the observed learned failure at acceptable cost, abandon a new-method claim. If the context channel is negligible, abandon this route as an important practical gap. Do not change a forecasting model's declared contract to manufacture failure.
''')
 put('research_ideas.md','''# Candidate triage

1. **Preserve dependence between persistent mechanism beliefs and evolving state.** A concrete native moment limitation is now verified. However, ordinary augmented-state inference corrects it, and is inexpensive in the tested native-scale inference case. A new covariance architecture is not justified. This remains an audit finding, not a recommended novel method.
2. **Calibrated reusable mechanism uncertainty from factual histories.** The broader cycle14 intake remains unresolved. Neural processes, context-conditional dynamics, HiP-RSSM, MTS3, persistent ensembles and belief-state models are essential controls. A new question needs a concrete residual those methods do not already solve. No new method is accepted.
3. **Coordinate-compatible reusable states.** Earlier native WorldTS evidence and direct causal-normalization prior art lowered this candidate's priority. No new evidence in cycle15 revives it.

No dataset-specific optimization is proposed. A general framework contribution needs an important unresolved mechanism and evidence beyond reproducing a known approximation or giving a standard method a new name.
''')
 put('final_research_direction.md','''# Cycle15 decision: a real native finding, no accepted new direction

**Research Question.** Does a probabilistic time-series world model preserve uncertainty from a persistent latent mechanism while propagating future state?

**Evidence.** A pinned native MTS3 prediction operator loses state/context cross-dependence in a prescribed admissible Gaussian process. At h64 with a=1,c=1,s=.5, its nominal95% marginal interval has true coverage19.55%. Closed-form, exact augmented-state propagation and independent50,000-particle simulations agree. This is a native operator finding; no trained-model or real-data failure is claimed.

**Existing Methods.** Standard joint Gaussian propagation corrects the witness. At native latent_obs_dim30,batch32,32-step inference takes about3.3ms and adds about5.3MiB of allocated memory in this test. PETS already retains model identity along rollouts. Therefore persistent sampling or a joint filter alone does not supply innovation.

**Proposed Idea.** No new architecture accepted. The present evidence does not justify specialized covariance compression or renaming the ordinary correction. At most, a trained-model audit could test the importance of this approximation; that audit is not itself an established top-venue research direction.

**Falsification.** A negligible learned context contribution, correct held-out calibration, or an adequate ordinary joint/particle correction would kill a new-method claim. Only a consequential residual under matched information, capacity and training/inference cost would motivate further design.

**Feasibility.** Pinned native source, executable cell and cost controls, original tensors and independent verification are preserved on the single5090 server. No training, processed-data substitution or unsafe pickle loading occurred. The source-native module test fits readily; full training cost is unmeasured.

**Next Experiment.** Establish learned-model relevance before designing a mechanism: inspect a faithfully trained hierarchical latent baseline's context coupling, then compare its declared forecast distribution with ordinary joint/persistent-sampling controls and independent held-out trajectories. Original data downloads were not completed and no full reproduction is claimed. If standard methods resolve the measured issue, switch direction.

**Final recommendation: none yet. Accepted novel directions:0.** The overall autonomous research goal remains active and unachieved.
''')
 status={'cycle':15,'native_operator_rows':630,'MC_horizon_rows':42,'MC_variance_intervals_verified':168,'ordinary_cost_rows':6,
         'new_training_fits':0,'native_trained_model_failure_established':False,'native_operator_discriminator_complete':True,
         'new_method_justified':False,'novel_direction_accepted':False,'objective_achieved':False,
         'decision':'Preserve native correlation-closure witness; ordinary joint inference is exact and practical at tested scale; do not claim new filter novelty',
         'data_acquisition':'Publisher processed Drive data unavailable in probes; optional original numeric fetch deliberately cancelled for cost-gate prioritization; no substitution or deserialization',
         'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 (ROOT/'outputs/task_status.json').write_text(json.dumps(status,indent=2),encoding='utf-8');print(json.dumps(status))
if __name__=='__main__':main()
