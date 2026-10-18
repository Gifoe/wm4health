from common89_v1 import *
import shutil

def markdown(name,text):
    p=C/name
    with p.open('x',encoding='utf-8') as f:f.write(text.strip()+'\n')

def main():
    assert not (O/'EVIDENCE_CLOSURE89_V1.json').exists()
    fit=read(O/'CSDI_PILOT89_V1.json');ev=read(O/'CSDI_EVAL89_V1.json');verify=read(O/'VERIFICATION89_V1.json')
    critic=read(O/'CRITIC_STRESS89_V1.json');ipf=read(O/'MARGINAL_CONTROL89_V1.json');oracle=read(O/'ORACLE_BELIEF_QUALIFICATION89_V1.json')
    mixture=read(O/'PERSISTENT_MIXTURE_FIT89_V1.json');failed=read(O/'PERSISTENT_MIXTURE_EVAL89_V1.json')
    assert failed['status']=='ORDINARY_BASELINE_ROLLOUT_FAILURE_NOT_EVIDENCE_OF_RESEARCH_GAP'
    assert ev['status']=='GENERATOR_NOT_QUALIFIED_DO_NOT_ATTRIBUTE_COUPLING_FAILURE'
    for run in ['csdi_train_v1','csdi_eval_v1']:assert read(O/run/'EXIT.json')['returncode']==0
    assert verify['full_goal_achieved'] is False and verify['novelty_verified'] is False
    maximum=max(r['projected_max_branch_tv'] for r in ipf['rows']);minimum=min(r['projected_coherence'] for r in ipf['rows']);converged=sum(r['converged'] for r in ipf['rows'])
    table='| System | Native CSDI energy | Exact mixture | Exact deterministic mean | Off-manifold ratio | Screen |\n|---|---:|---:|---:|---:|---|\n'
    for r in ev['family_summary']:
        table+=f"| {'Duffing' if r['family']==0 else 'Reservoir'} | {r['energy_score']:.9f} | {r['exact_population_energy_score']:.9f} | {r['deterministic_mean_energy_score']:.9f} | {r['off_manifold_ratio']:.6f} | FAIL |\n"
    markdown('literature_review.md', '''# Literature and novelty boundaries

The broad problem of jointly plausible action branches is occupied. [OneWorld](https://arxiv.org/abs/2609.30946) learns shared-mechanism compatibility across action-conditioned video futures. Its prior-corrected exact evidence-ratio construction can recover a shared joint distribution with correct marginals under its assumptions. The learned compatibility training objective and the native flow guidance must be distinguished from terminal Gibbs reweighting of an oracle-ID critic. No released native OneWorld model was run here.

[PAWBench](https://arxiv.org/abs/2608.27345) studies probabilities of repeated futures under a fixed initial observation/action. [EDDY](https://arxiv.org/abs/2605.06553) studies marginal-preserving diffusion-sample interaction, and [Couple to Control](https://arxiv.org/abs/2605.11311) studies initial-noise couplings. [OSCC](https://arxiv.org/abs/2609.33543) explicitly constrains observation-safe, marginal-preserving branch coupling for policy-gradient estimation in imperfect-information games. Marginal preservation, common random numbers, Bayesian mixtures, IPF and optimal transport are not new primitives.

The remaining working intersection is probabilistic coherence of *different hypothetical action queries* in learned time-series world models. Adding plans supplies no new real observation. Single-query marginal preservation alone is weaker than full projective consistency of all overlapping branch subsets. Source-scope differences do not establish novelty, and this cycle does not claim a verified new problem.

[CSDI](https://arxiv.org/abs/2107.03502) is a probabilistic time-series imputation/forecasting model. This cycle transfers its native conditional diffusion core to controlled action-conditioned trajectories; it does not reproduce its original benchmark tables. Native source commit: `ermongroup/CSDI@7f24a436f08d98853a6b43d4f7f04e5a65ecdf27`.

[DALI](https://arxiv.org/abs/2508.20294) and [Meta-iLaD](https://proceedings.mlr.press/v306/ye26p.html) already concern contextual/identifiable latent dynamics. Source inspection is feasibility evidence, not model reproduction. The DALI RSSM does pass supplied context into its transition; an earlier tentative ignored-context suspicion is not supported by source.

Final boundary: no new architecture or theorem is justified by the present evidence. A competent stochastic model plus closest existing coupling methods is still required before a method contribution can be argued.
''')
    markdown('baseline_registry.md',f'''# Models, data, source and reproduction status

| Entry | Exact status | Scope |
|---|---|---|
| Native CSDI | {fit['parameters']} parameters, one seed, {fit['updates']} updates, selected step{fit['selected_step']} | Cross-task transfer; both generator competence screens fail |
| Gaussian-product critics | Six finite-ID fits, three seeds, two negative laws | Privileged mechanism-ID stress; no stochastic world-model reproduction |
| Count/DeepSets critic | Three finite-ID fits | Ordinary flexible density-ratio comparator; capacity differs |
| IPF | 27 exact finite-support replay cells | Established postprocessing control; {converged}/27 meet strict1e-10 tolerance |
| Polynomial persistent mixture | Two family-specific40-iteration EM fits | Ordinary unsupervised transition-mixture control; first future-rollout family overflows |
| Exact Bayes mixture | Two nonlinear families | Analytic evaluator/control, not a learned method |
| DALI |105 pinned text files; no training/weights | Source interface audit only |

The CSDI TRAIN split has8192 trajectories; CAL256; initial EVAL64 action queries in16 independent four-plan groups. There is no observation noise, two fully observed state variables, one external control, and one known family indicator. A persistent nine-point mechanism remains uncertain under identical zero context. These are intentionally small controlled systems, not real-world evidence.

TRAIN-only state scaling, seeds, simulator hash, native source hashes, data/checkpoint hashes and raw server-side payload locations are recorded in the JSON receipts. Native CSDI source files remain unchanged. All139 gradient tensors checked were finite; masked-future dummy changes leave sampled targets exactly unchanged. No optional import sentinel was actually needed. See the frozen protocols for all settings and for architecture/input-information differences.

Failed polynomial control: fitting reads observed states/actions and family only, not hidden mechanism IDs. Known component count and polynomial Markov bias are additional inductive assumptions. Teacher-forced CAL likelihood selected the checkpoint; that did not qualify closed-loop rollout stability. Do not treat this failed baseline as a strong model defeated by a proposed method. The fresh-query evaluation stopped before CSDI re-evaluation, so there is no fresh matched comparison.
''')
    markdown('experimental_results.md',f'''# Experimental results

No publishable direction is accepted. The following are measured results with different scopes.

## Exact Bayesian metric qualification

Under identical uninformative histories, the correct posterior mean of two nonlinear physical worlds need not belong to any single-parameter physical trajectory family. An exact two-world mixture matches with zero error in both systems; a single-world parameter fit has positive held-out error. Consequently, single-world membership is not a valid universal test of predictive *means*. This does not refute sample-level shared-world evaluation.

## Learned compatibility-head stress and ordinary control

Nine fitted critics use oracle mechanism IDs, not raw time-series inputs. Terminal Gibbs recoupling can have nearly perfect same-mechanism agreement and large marginal distortion. The exact shared/independent Bayes density ratio is an explicit successful control.

Ordinary IPF replay covers27 cells (K2/3/4). Maximum projected branch TV is {maximum:.12g}; minimum shared-mechanism agreement is {minimum:.12g}. All nine original selected-checkpoint population BCE values replay exactly. Only{converged}/27 cells reach the registered1e-10 maximum-coordinate marginal tolerance within10000 sweeps;18 do not. Report residuals rather than claiming exact convergence. All nine distinct-negative cells reach tolerance in one sweep. This strongly weakens a new-method claim based only on the finite stress.

## Actual native stochastic time-series generator

CSDI trained for{fit['updates']} updates in{fit['elapsed_seconds']:.3f} seconds, one optimizer seed, one mixed-family model. Selection used CAL denoising loss only. Selected checkpoint CAL loss={fit['cal_denoising_mse']:.9f}. EVAL uses128 samples per action query; scoring integrates over the exact nine-point target distribution.

{table}

Lower energy is better. Each family improves over the deterministic posterior mean, but both exceed the prospectively declared.5 physical residual ratio threshold. This means the pilot is not qualified for a mechanism-coupling failure claim. Eight query groups per family support descriptive grouped bootstrap intervals; they do not replace independent optimizer seeds.

## Ordinary persistent-latent transition model

Two EM fits completed with CAL-selected iterations{mixture['families'][0]['selected_iteration']} and{mixture['families'][1]['selected_iteration']}. Fresh held-out rollout failed in family{failed['failure_family']}: `{failed['error']}`. No samples were dropped and no EVAL-based repair was made. No fresh matched metric table or successful coupled-generation result exists.

## Verification

Independent Gram-matrix distance replay recalculated{verify['metric_values_recomputed']} score values from the server raw sample arrays; maximum error={verify['independent_gram_vs_cdist_max_abs']:.12g}. Native source, data and checkpoint hashes and CAL checkpoint selection were verified. An analytic two-point energy-score control passed. Verification confirms these bounded results, not scientific novelty or completion of the research objective.
''')
    markdown('negative_evidence.md',f'''# Negative evidence and stopped claims

1. A posterior mean may be outside the physical trajectory manifold even when its underlying stochastic world mixture is correct. Do not call this a mechanism-learning failure without qualifying what the model outputs.
2. Compatibility classification alone does not identify the shared/independent density ratio. The finite-ID Gibbs stress is not native OneWorld flow guidance and cannot establish a released-model failure.
3. Existing marginal-constrained projection nearly removes the entire finite marginal distortion while retaining high agreement (max TV{maximum:.6g}, min agreement{minimum:.6%}). Strict tolerance passes only{converged}/27 cells. This is evidence against claiming a new coupling architecture from that toy diagnostic.
4. Native CSDI's3000-update transfer improves energy over exact deterministic means but fails both physical-support screens. Training sufficiency, transfer protocol and model inductive bias remain plausible explanations. Do not promote the result to a general limitation of trained time-series world models.
5. The ordinary degree5 mixture control overflows in free-running EVAL despite strong teacher-forced CAL likelihood. This disqualifies it as a successful strong baseline. It also exposes a missing CAL rollout qualification in this control pipeline; no test-tuned repair was performed.
6. The proposed multi-action question overlaps substantial existing work. Neither distinct conditioning nor time-series application alone establishes novelty.

No inference is made that all ordinary persistent-latent models fail. No additional CSDI epochs, polynomial clipping, dropped components or hyperparameter variants were selected using these EVAL results.
''')
    markdown('hypothesis_space.md','''# Competing explanations and tests

| Hypothesis | Current evidence | Next discriminating evidence |
|---|---|---|
| A correct uncertain physical posterior must have a mean in one-world trajectory space | Falsified by exact two-world simulation in two systems | Use distribution/sample-aware evaluation |
| Good relational compatibility classification ensures calibrated Gibbs recoupling | Falsified in finite oracle-ID diagnostics | Proper density-ratio controls already succeed ideally |
| A new coupling architecture is needed for the finite setting | Strongly weakened by ordinary Bayes/IPF controls | Do not develop a new architecture for this toy alone |
| Learned stochastic time-series generators show a practically important query-list probability defect | Unresolved; no qualified learned-generator coupling test completed | Competent frozen generator plus closest existing coupling controls and risk/decision outcomes |
| CSDI's weak physical support is specifically a coupling defect | Unsupported: no coupling was applied | First qualify generator training and support on CAL with independently frozen EVAL |
| The fitted ordinary polynomial mixture is a sufficient strong baseline | Rejected as executed: free-running rollout overflows | A separately specified bounded/native persistent-latent model with CAL rollout screening |
| Full cross-query projective consistency follows from preserving each singleton marginal | False as a general statement; no new theorem claimed | Evaluate overlapping branch-subset laws, with shared-latent Bayesian controls |

Do not use the lack of a successful strong baseline as proof that a new mechanism is needed. Future experiments must identify a meaningful constraint that known coupling/generative constructions do not already solve.
''')
    markdown('research_ideas.md','''# Research candidate triage

## Candidate1: probabilistically coherent multi-action time-series simulation

Question: can action-conditioned futures share a physical mechanism without changing the original predictive probabilities when more hypothetical plans are queried?

Importance: planners compare possible actions under uncertainty; imaginary query lists are not additional real evidence. Risk estimates should not change merely from adding alternatives. This is a general world-model property, not optimization of one dataset.

Current status: **working evaluation candidate; insufficient evidence for acceptance**. Broad shared-physics generation and marginal-preserving coupling are occupied. The finite diagnostic is largely repaired by standard methods, and the actual CSDI transfer is not physically qualified. No new method is recommended yet. A full new-architecture claim on current evidence should be retired.

Possible contribution only if supported later: a differentiated, practically important failure of qualified learned time-series world models under multiple action queries, with a minimal correction that survives ordinary Bayesian/shared-latent/marginal-constrained controls. Naming a known coupling primitive differently would not count.

No second or third candidate is promoted merely to fill a list. The complete research objective remains unachieved.
''')
    markdown('final_research_direction.md','''# Research decision: no final direction accepted

**Research Question:** Can a learned time-series world model provide physically consistent distributions over multiple hypothetical action branches while preserving per-action probabilities and overlapping-query consistency?

**Evidence:** Finite privileged-ID critic diagnostics show that relational agreement and probability calibration differ. Exact Bayes and ordinary IPF provide strong successful controls. An actual native CSDI model was trained and evaluated, but it failed prospective physical-support screening. An ordinary learned polynomial transition mixture failed free-running evaluation. These facts do not yet establish a publishable unresolved world-model problem.

**Existing Methods:** Shared physical mechanisms, marginal-preserving sample interactions, latent-variable mixtures and observation-safe couplings have substantial prior art; see literature_review.md. The present finite experiment does not demonstrate their insufficiency.

**Proposed Idea:** None accepted as a new architecture. Retain the multi-action probability question as a working evaluation hypothesis only. Do not rename Bayesian coupling or IPF as innovation.

**Falsification:** If an existing shared-latent or calibrated density-ratio/marginal-constrained construction provides accurate physical distributions and stable decision/risk estimates with practical cost, abandon a new-method claim for that setting. If qualified models do not show query-induced distortion, abandon the empirical failure hypothesis as well.

**Feasibility:** The pinned native CSDI core runs on the server RTX5090; the pilot needs under three minutes and small GPU memory. Existing data/simulator, raw samples, selected checkpoints and original code remain server-side. More compute does not substitute for a meaningful unsupported property. Known low-dimensional Markov dynamics make the current synthetic setting a feasibility test, not a final benchmark.

**Next Experiment:** Before any new mechanism, qualify a frozen stochastic world-model generator and an ordinary persistent-latent control on CAL closed-loop physical support/calibration, then use fresh independent EVAL queries to compare native independent generation, exact Bayesian/shared-latent coupling and a closest existing marginal-preserving method. Measure both singleton probabilities and overlapping branch-subset laws, plus risk/plan-choice consequences. Do not repeat the failed CSDI/polynomial settings with EVAL-selected hyperparameters. A real training-qualified failure that standard controls cannot remove is the decision point.

**Recommendation:** One candidate remains under evaluation; accepted final directions=0. Research goal ACTIVE and UNACHIEVED. The current evidence is insufficient for a novelty or top-conference-potential claim.
''')
    before=ROOT/'CURRENT_RESEARCH_STATUS.md';archive=ROOT/'CURRENT_RESEARCH_STATUS_BEFORE89_V1.md'
    assert not archive.exists();shutil.copyfile(before,archive)
    status=f'''# Current time-series world-model research status

C89 closure: actual native stochastic time-series transfer, finite compatibility stress and ordinary falsification controls. **Research goal ACTIVE/UNACHIEVED; accepted final directions0.**

Nine critic fits do not constitute nine world models. Twenty-seven standard IPF cells retain at least{minimum:.6%} mechanism agreement with maximum branch TV{maximum:.6g}; only{converged} meet strict tolerance. This weakens the new-method justification.

One native CSDI model completed3000 updates and64-query distributional evaluation. It beats the deterministic posterior mean on energy in both systems but fails both physical-support competence screens. No learned-generator coupling failure claim is supported. Two ordinary EM transition-mixture fits completed, but future rollout overflows; no successful strong-model comparison is claimed.

Candidate: probability-preserving and physically consistent multi-action simulation. Generic novelty is unverified; no final paper direction or architecture is accepted. The next scientific test requires qualified learned generators and closest existing coupling controls. See cycle89/final_research_direction.md and all negative evidence.

Original C88 action-timing candidate remains retired. All earlier work, independent long-running C71 campaign and unrelated experiments are preserved. Raw data/checkpoints/sample arrays remain only on the server storage volume; publication contains authored source, protocols, bounded metrics and reports.
'''
    with (ROOT/'CURRENT_RESEARCH_STATUS_V89.md').open('x',encoding='utf-8') as f:f.write(status)
    before.write_text(status,encoding='utf-8')
    receipt=dict(utc=utc(),status='C89_BOUNDED_EVIDENCE_CLOSED_NO_ACCEPTED_FINAL_DIRECTION',learned_native_stochastic_models_trained=1,ordinary_family_mixture_models_fitted=2,critic_heads_fitted=9,strictly_converged_ipf_cells=converged,ipf_cells=27,ipf_maximum_projected_tv=maximum,ipf_minimum_mechanism_agreement=minimum,generator_competence_passed=False,ordinary_mixture_rollout_failed=True,full_goal_achieved=False,accepted_research_directions=0,reports=[p.name for p in [C/'literature_review.md',C/'baseline_registry.md',C/'experimental_results.md',C/'negative_evidence.md',C/'hypothesis_space.md',C/'research_ideas.md',C/'final_research_direction.md']],receipts={name:sha(O/name) for name in ['CSDI_PILOT89_V1.json','CSDI_EVAL89_V1.json','VERIFICATION89_V1.json','MARGINAL_CONTROL89_V1.json','PERSISTENT_MIXTURE_FIT89_V1.json','PERSISTENT_MIXTURE_EVAL89_V1.json']},previous_canonical_sha256=sha(archive),current_canonical_sha256=sha(before))
    write(O/'EVIDENCE_CLOSURE89_V1.json',receipt);log(dict(stage='c89_evidence_closed',status=receipt['status'],accepted_research_directions=0))

if __name__=='__main__':main()
