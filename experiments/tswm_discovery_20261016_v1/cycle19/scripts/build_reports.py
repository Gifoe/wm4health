"""Generate cycle-local evidence reports from completed scientific jobs."""
import hashlib,json,pathlib
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8'))
event=read('outputs/event_audit_attempt1/results.json');attribution=read('outputs/attribution/results.json');sid=read('outputs/system_id_control/results.json')
checks=[read('outputs/event_audit_attempt1/independent_verification.json'),read('outputs/attribution/independent_verification.json'),read('outputs/system_id_control/independent_verification.json')]
assert all(x['pass'] for x in checks)
native=json.loads((ROOT.parent/'cycle18/outputs/native_demo/results.json').read_text())
def write(name,body):(ROOT/name).write_text(body.strip()+'\n',encoding='utf-8')
table=['| Task | D_stsp | D_H | MASE10, true denominator | Rank event-CDF difference at100 |',
       '|---|---:|---:|---:|---:|']
for row in sid['rows']:
    m=row['global_metrics'];d=row['event_comparison']['own_rank_diagnostic_all_offsets']['CDF_difference'][2]
    table.append(f"| {row['task']} | {m['D_stsp']:.6f} | {m['D_H']:.6f} | {m['MASE_true_denominator']:.6f} | {d:+.6f} |")
table='\n'.join(table)
write('literature_review.md','''# Literature and novelty boundary: cycle19

The target is a reusable time-series world model, not optimizing one dataset. Path-event fidelity was screened as a candidate capability. It has not passed the novelty/remaining-failure gate.

- [DynaMix, NeurIPS2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/1419d8554191a65ea4f2d8e1057973e4-Abstract-Conference.html) is the executed autonomous pretrained baseline. Its original released examples were reproduced in cycle18; this cycle retains those arrays and probes path laws. Do not impose a Bayesian-posterior interpretation on its native exploration noise.
- [DynaBase,2607.14937v1](https://arxiv.org/html/2607.14937v1) already reduces context-conditioned reconstruction to a two-parameter affine nearest-neighbor map, studies context parroting versus chaotic behavior, and contrasts predictive and reconstruction training. This is a strong boundary against claiming nearest-neighbor dynamics, compact architecture, or the broad prediction-versus-simulation distinction as new. Primary methods/theory/limitations read; official implementation and aggregate results were not executed here. The eight-neighbor resampling control is not this paper's algorithm.
- [SINDy,PNAS2016](https://doi.org/10.1073/pnas.1517384113) already discovers governing equations with a candidate library and sparsity. Our dense ridge polynomial/hinge fit is an ordinary custom control, not a native SINDy reproduction or a new discovery algorithm.
- [VAMPnets,Nature Communications2018](https://www.nature.com/articles/s41467-017-02388-1) already learns kinetic representations and transition models. First-passage, transition kinetics, and Markov/Koopman consistency are established concepts. No VAMPnet implementation was trained in this cycle.
- [Learning stochastic dynamics and predicting emergent behavior using transformers,2024](https://www.nature.com/articles/s41467-024-45629-w) already studies learning transition rates and emergent behavior from trajectories. No source execution here.
- [Statistically accurate dynamical learning,NeurIPS2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/4dc57702c987e1e72f0dd2921edb5ded-Paper-Conference.pdf) provides an ergodic generalization formulation and Jacobian-informed learning. Screening its primary abstract/theory blocks a broad claim that physical statistics versus prediction error or Jacobian regularization is unexplored. Its guarantees are conditional, not universal. No code execution here.
- [Topological OOD generalization,2606.22969v1](https://arxiv.org/html/2606.22969v1) already targets vector-field topology and feature splitting; [DSR adequacy test,2606.24873v1](https://arxiv.org/html/2606.24873v1) discusses alignment/reliability and stochastic degeneracy. Archived/read, not executed. Their existence does not prove all path laws correct, but generic topology/reliability slogans are insufficient differentiation.
- [World models under asynchronous observations,2609.07299v1](https://arxiv.org/html/2609.07299v1) distinguishes sensing schedules from refresh-gated actuation and tests staleness/time-to-refresh. Primary method/results/limitations inspected; no native reproduction here. Generic asynchronous-input awareness is not an untouched gap.
- [TSWM accuracy/mechanism consistency,2610.01842v1](https://arxiv.org/html/2610.01842v1) already compares forecasting architectures and declared directional responses on eight action datasets. Its direction score is a limited property, not full causal identification. Primary formalization read; current cycle does not reproduce its dataset results. Broad action-sensitivity issues remain covered by cycle7's closest-work ledger.
- [KITE,2610.12738v1](https://arxiv.org/html/2610.12738v1) was screened and archived as a recent event/knowledge-conditioning neighbor. No method-performance conclusion or source execution is claimed.

Versioned primary HTML and SHA256 registries are retained under outputs/literature. Prior literature from cycle18 and earlier remains authoritative for those cycles. This is a targeted screening ledger, not an exhaustive survey or a venue-quality score.
''')
write('baseline_registry.md','''# Baselines and execution scope

| Object | Execution | Role and limitations |
|---|---|---|
| DynaMix Python39cc55514a4c5a0c50534c092fd9c3d40abdabc8, released3D model | cycle18 nine native10000-step draws; retained here | Frozen10256-parameter autonomous model; three public examples, not paper54-system aggregate |
| Historical context replay | Retained; scored on path events | Descriptive memory control, not a learned universal model |
| Eight-neighbor successor resampling | Three seeds on each original context | Old analogue idea; strict past-only successors; not DynaBase or an exhaustive strong baseline |
| Native attention sigma=0 | Three original systems, one deterministic draw each | In-memory ablation; all other checkpoint tensors/disk bytes unchanged |
| Common-origin Chua native history/normalization wrappers | Three conditions x three stochastic draws | Same future and reference thresholds; more history is additional information, not a fair method win |
| Dense ridge vector-field library | Five past-only fits,12 validation candidates each;10000-step autonomous rollouts | Classical per-system fit, degree2/3 polynomial plus optional decile hinges, four-substep RK4; no known plant equations; different training budget from pretrained model |
| DynaBase/VAMPnets/transformer transition-rate method/Jacobian training | Literature only | Their absence from execution prevents a claim that existing methods are insufficient |

Runtime: native/model-control jobs use server E:/Anaconda/python.exe3.13.9,torch2.11.0+cu128,numpy2.3.5,CPU,process-local single thread. The retained event audit and its index verifier use the existing project .venv Python3.10 with numpy1.26.4; command paths are captured per job. RTX5090 was not needed for these short inference/control jobs. No package installation or mutation of either environment. Ordinary regression is new fitting; there is no new neural-network training.

Scientific job statuses, stdout, stderr and exact entry-script snapshots are under outputs/jobs. Plans distinguish original statistics, post-failure all-offset diagnostics, and later exploratory attribution/control fits. No public publishing or repository push.
''')
write('experimental_results.md',f'''# Completed cycle19: exploratory path-law audit with falsification controls

The data support a localized event-law discrepancy in one released Chua example. They do not establish an unsolved general world-model problem. Ordinary fitting substantially weakens architecture necessity.

## Protocol and estimand

Three original released systems, context lengths2000/1028/512, future10000 sample indices. Reuse all nine native cycle18 draws. Coordinate1 event: start below context median and hit its context90th percentile within10/50/100 samples, including the start. Primary landmarks have stride100 inside500-sample blocks. The all-offset supplement uses401 full100-sample windows per block; it was added after the first scoring attempt failed on a zero denominator. Rank diagnostics use each full path's own evaluation quantiles, not deployable model calibration. No physical sampling interval is inferred.

Block bootstrap intervals are descriptive: one reference per system, no verified mixing/coverage, and stochastic draws are not independent physical/training replications. Completed-return CDF is secondary and censor-biased. Empty conditional halves remain null with eligible counts, never a zero probability.

## Pilot and attribution

- Original Chua native all-offset CDF differences at100 are+0.1803 to+0.2243 with fixed context thresholds; rank differences+0.1491 to+0.1838. Reference rank CDF is0.392956. These are exploratory fixture-specific estimates.
- Sprott's coarse-grid apparent discrepancies shrink to rank max errors0.0120–0.0141 under all offsets, showing serious phase/grid sensitivity. The reference second-half primary cohort has zero eligible landmarks. Do not call the original31–55pp coarse discrepancy a physical failure.
- Lorenz rank max errors vary0.0549–0.1606 across native draws. There is no uniform three-system path-law failure.
- Zeroing native attention noise worsens geometry on all three systems: D_stsp1.92765/7.99066/7.39117 versus native means0.78178/2.03132/3.17179. It improves point MASE but does not repair Chua path fidelity. Exploration is purposeful in the released model.
- At common Chua origin2000, longer context improves global geometry, including when the original short-context normalization is held fixed. Rank event residuals depend on origin/history; the fixed-coordinate short wrapper reproduces the native64-step output exactly. This is attribution on the same fixture, not independent confirmation.

## Ordinary vector-field control

Twelve specifications per task are selected only on past64-step validation rollouts, then refitted on all available context. No future-based specification changes. All five10000-step rollouts remain finite. Native metrics below use the retained official metric implementation; MASE uses the true series denominator.

{table}

Original-system library geometry0.3213/1.0633/0.3250 and spectra0.0403/0.0414/0.0281 improve on native means for every demonstration. At common Chua origin, the long-history library fit reduces rank-CDF difference at100 from0.1675 in its short-history fit to0.02284; its geometry improves1.5326 to0.6239. The short-history library fit is worse than native for that event statistic. History and ordinary model choice matter; no blanket superiority or universal repair is claimed. Per-context fitting and a frozen pretrained model have different budgets.

## Verification and failures

The first event job exited1 during strict JSON serialization of an empty conditional half. Its computed analogue arrays were retained. Attempt1 serializes unavailable values as null; all three control arrays are byte-identical to the failed attempt.

Independent event indexing verifies96 retained-path cohorts,90000 past-only analogue transitions, zero event-count differences and maximum reversal-moment arithmetic error8.33e-17. Attribution verifies12 array hashes,48 event cohorts and six exact native64-step prefix replays. The ordinary control verifies five context-only selection/refit replays,20 independently counted event cohorts and exact64-step replay. Linear-flow integration max error7.56e-11; regression coefficient error2.78e-17; largest ridge normal-equation residual1.88e-16. These checks establish arithmetic/provenance, not novelty or physical uncertainty coverage.

Attribution scientific job146.37s; ordinary control33.48s wall time. Existing env/source/checkpoint unchanged. All23 new raw trajectory NPZ files (including failed/retry duplicates) are mirrored and SHA256 compared across hosts, total5706190 bytes.

The three-state time-reversal witness has identical scalar autocovariances but different directed first-step probabilities0.8/0.1 and MFPT1.36986/2.32877. Ordinary transition likelihood already distinguishes it (KL1.45561). This classical witness checks metric insufficiency; it is not a new theorem, benchmark contribution or method.

![Exploratory controls](outputs/analysis/event_law_controls.png)

Decision: no new architecture or accepted research direction. Stop optimizing these three already inspected test fixtures. Original primary statistics and post-pilot analyses remain separately labeled.
''')
write('negative_evidence.md','''# Negative evidence and claims rejected in cycle19

1. **Uniform event-law failure of modern autonomous models:** unsupported. Sprott's large coarse-grid result is strongly estimator/phase-sensitive; Lorenz is heterogeneous; only Chua has a clear exploratory residual across the three draws.
2. **A new kinetic-fidelity concept:** rejected as a novelty claim. Markov/VAMP/transition-path work already studies kinetics; scalar stationary metrics being insufficient is old.
3. **Remove exploration noise to repair simulation:** rejected here. Geometry deteriorates for all three systems and the Chua event residual worsens, despite better short point forecasts.
4. **Normalization alone explains everything:** unsupported. Holding short-context coordinates fixed still allows substantial geometry gains from longer history, and rank event residuals do not vanish uniformly.
5. **A sophisticated new world-model architecture is necessary:** not established. A small classical library fit selected on past data improves global scores on all three examples, and longer context reduces the common-origin Chua event residual substantially. Its short-context event result is also worse in one condition, so do not claim a universal classical solution.
6. **Three random forecast draws demonstrate cross-system statistical significance:** rejected. They share one checkpoint, one context and one future reference per system.
7. **This experiment validates a universal framework:** rejected. There is no new framework here, and three small released fully observed examples do not establish broad deployment capability.
8. **Another forecast-versus-dynamics or Jacobian-loss idea is novel:** rejected at that level of abstraction. DynaBase, DPC/KAFFEE, NeurIPS2024 statistical accuracy and established response theory already provide direct precedents.

No test-set hyperparameter search follows the reported ordinary-control results. Failed attempt logs/arrays, unhelpful controls and all validation candidates are retained. Overall research-discovery goal is still unachieved.
''')
write('hypothesis_space.md','''# Competing hypotheses after executed controls

| Explanation | Discriminating evidence | Current judgment |
|---|---|---|
| Global native metrics hide an ordered path residual | Frozen-path event law, rank diagnostic, ordinary controls | Local exploratory Chua residual observed; broad failure not established |
| Sampling-grid resonance creates apparent path failure | Preserved stride100 statistic versus all-offset supplement | Strong explanation for Sprott; supplement is post-pilot, not confirmation |
| Native exploration alone creates excess crossings | Zero only sigma across all original systems | Simple noise removal fails globally and for Chua event law |
| Context history/normalization dominates | Common origin, same future, fixed-short-coordinate and native-long wrappers | History clearly matters; normalization alone not sufficient |
| Ordinary per-context system identification suffices | Past-only selected polynomial/hinge ridge, all three systems and history arms | Strong competing control; no architecture necessity established |
| Remaining residual is reference variation/finite-context ambiguity | Independent longer source trajectories, matched-context competitors | Still unresolved; these references alone cannot settle it |
| Event-law/response fidelity supplies a new generic concept | Closest primary literature and source review | No: well-established concepts and directly relevant recent work |

A residual error is not evidence that all existing methods fail. Any renewed path-law method must first survive independent system/trajectory replication and competent generative/MSM/VAMP/DynaBase controls with matched available inputs. Do not tune a proposed method on these already inspected future fixtures.
''')
write('research_ideas.md','''# Candidate triage: no accepted idea

**Current priority is a general time-series world-model capability, following the user's scope correction. Dataset-specific fitting is only a falsification control.**

Candidate investigated: preserve ordered path-event laws in a reusable context-conditioned simulator. Important for simulation, but generic novelty is already covered by kinetic modeling and existing reconstruction/response literature. Empirical evidence is currently one localized Chua residual and substantial ordinary-control improvements. **Do not recommend a new path-event loss or architecture from this evidence.**

Two adjacent possibilities were screened, not promoted: (i) context retrieval versus transferable dynamics, already directly addressed by DynaBase and context-parroting literature; (ii) local perturbation/response fidelity, already addressed in Jacobian-informed learning and response theory. Neither is a validated new research question merely because current public models omit a diagnostic.

Maximum three final recommendations is respected: zero pass the evidence and novelty gate. This does not prove every research problem solved. It means this particular evidence does not yet justify one. The next discovery step should compare precise residual capabilities of existing general models, rather than add a dataset, increase epochs, or rebrand an established regularizer.
''')
write('final_research_direction.md','''# Research status: general time-series world model; no validated new direction yet

**Research Question:** Can a reusable time-series world model preserve task-relevant evolution laws across systems beyond what ordinary contextual dynamics and existing probabilistic/state-space methods already provide? This is the selection objective, not a novel research claim by itself.

**Evidence:** Cycle19 completes an actual native-trajectory event audit,12 new native ablation rollouts and five classical vector-field fits on the server. Chua shows an exploratory first-passage residual; Sprott's initial anomaly is highly grid-sensitive. Ordinary past-only library fitting improves global geometry/spectrum on all three released examples and reduces common-origin long-history Chua rank event bias to0.02284. Independent arithmetic, hash, selection and native-prefix checks pass. No unsupported data ordering, future-trained selection, or fabricated physical time step is used.

**Existing Methods:** DynaMix already provides autonomous cross-system reconstruction; DynaBase exposes a compact contextual construction; VAMP/Markov methods learn kinetics; ordinary system identification is competitive; statistical-accuracy/Jacobian and response theory already study stronger dynamical fidelity. These references block generic novelty claims, but do not imply a narrowly specified remaining problem is impossible.

**Proposed Idea:** None accepted. The executed controls do not justify another event-loss, noise-removal, memory or generic consistency architecture. The target remains a general framework, with multiple systems serving as evidence rather than a single-data optimization result.

**Falsification:** Reject architecture necessity if competent contextual identification/kinetic/generative models remove the residual under matched available information. Reject the phenomenon if it disappears across independent references or an estimator correction. Reject innovation if the same mechanism and protocol are already supplied by the closest literature. Several of these concerns are already supported in this cycle.

**Feasibility:** All current jobs run quickly on existing CPU software; the single RTX5090 remains available for a justified method test. Source/checkpoints/data/configs/logs/plans/raw arrays are preserved on both hosts. These three native demonstrations are sufficient to falsify weak ideas, insufficient to claim a universal new framework.

**Next Experiment:** Do not continue tuning these inspected fixtures. First identify a precise capability residual beyond DynaBase, competent state-space/generative models, Jacobian-informed learning and response theory; require an independent held-out system family and an ordinary control before designing a method. A transient/perturbation test requires legitimate generator/observation provenance and must distinguish state intervention from a change to the physical vector field. This is a research-selection gate, not a claim that such a gap already exists.

**Recommendation:** zero directions currently meet the complete novelty/evidence/solvability standard. Cycle19 is complete as an audit; the overall autonomous research goal remains ACTIVE and UNACHIEVED. No public release or push was performed.
''')
summary={'cycle':19,'accepted_novel_directions':0,'objective_achieved':False,'cycle_status':'completed_exploratory_audit',
    'decision':'NO_ARCHITECTURE_NECESSITY_ESTABLISHED; STOP_TUNING_INSPECTED_FIXTURES',
    'system_id_table':[{k:r[k] for k in ['task','array_sha256','global_metrics']} for r in sid['rows']],
    'verification_receipts_pass':[x['pass'] for x in checks],
    'source_result_hashes':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
       ['outputs/event_audit_attempt1/results.json','outputs/attribution/results.json','outputs/system_id_control/results.json']}}
(ROOT/'outputs/research_decision.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps({'reports_generated':7,'accepted_novel_directions':0}))
