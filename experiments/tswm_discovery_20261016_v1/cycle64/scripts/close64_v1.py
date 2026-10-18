"""Render evidence reports on the server; no local result synchronization."""
from common64 import *
from collections import Counter
import numpy as np,scipy,torch,platform,sys
f=read(OUT/'FIT_AND_GAUSSIAN_EVALUATION64_V1.json');s=read(OUT/'SAMPLER_EVALUATION64_V1.json');c=read(OUT/'CAUSAL_FACTORIZATION64_V1.json');a=read(OUT/'INDEPENDENT_REPLAY64_V1.json');i=read(OUT/'IID_REFERENCE64_V1.json');summary=read(OUT/'SUMMARY64_V1.json');q=read(OUT/'QUALIFICATION64_V1.json');sources=source_check();assert a['status']=='PASS' and i['status']=='PASS'
reports={}
def save(name,text):
 p=ROOT/name
 with p.open('x',encoding='utf-8') as out:out.write(text.strip()+'\n')
 reports[name]=sha(p)
def table(headers,rows):return '| '+' | '.join(headers)+' |\n|'+ '|'.join(['---']*len(headers))+'|\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in rows)
save('README.md','''# C64: frozen observational trajectory reuse under intervention

Status: CLOSED GAUSSIAN ATTRIBUTION. Overall research goal ACTIVE and UNACHIEVED. Accepted final research directions: 0.

This cycle executed train-only Gaussian MLEs and oracle sampler attribution on a fully observed feedback system. It is not a neural diffusion training run, real-data experiment, or full reproduction of TDS. The classical causal AR baseline is substantially stronger than direct block conditioning. A retained full joint Gaussian also supports normalized temporal causal factorization without retraining. No new framework is validated.

Read `experimental_results.md` for all fixed comparisons and `negative_evidence.md` for rejected interpretations. `final_research_direction.md` specifies the remaining provisional test and abandonment criteria. The seven requested reports, machine-readable outputs, immutable scripts and protocol are included. Raw arrays/checkpoints stay on the authorized server storage, with paths and SHA256 receipts.

Reproduce on a separately provisioned environment with Python 3.10, NumPy, SciPy and CPU PyTorch: set the project storage registry; fetch the three pinned upstream utility files listed in `outputs/SOURCES64_V1.json` and verify hashes; run `qualify64_v1.py`, then `fit64_v2.py`, `samplers64_v1.py`, `causalize_joint64_v1.py`, `replay64_v3.py`, `iid_reference64_v1.py`, `close64_v1.py`. Outputs are deliberately exclusive-create. Use a fresh output directory/storage registry for an independent run. The original upstream files are not redistributed because a root license was not located. Their receipt includes pinned links and checksums.

Operational version history: `fit64_v1.py` was not executed; its normal-equation helper is unsuitable at H1. Actual fitting uses SVD in v2. Replay v1 failed parsing before execution; replay v2 stopped at the exactly rank-deficient H1 QR check and produced no scientific output. v3 fixes the QR rank cutoff at 1e-12, compares predictions rather than nonidentified coefficients, and passes. No data, learned model, query, sampler budget, precision floor or scientific metric changed in these repairs.
''')
save('literature_review.md','''# Literature and novelty boundary, 18 October 2026

The original conditioning-versus-intervention distinction is classical and was already reviewed in C29. C64 does not claim that insight, sequential g-computation, inverse policy weighting, Gaussian causal factorization, diffusion guidance or SMC as new.

| Primary source | Relevant existing capability | Scope and current status |
|---|---|---|
| [CaTSG, arXiv:2509.20846v3](https://arxiv.org/html/2509.20846) and [author code](https://github.com/microsoft/TimeCraft/tree/main/CaTSG) | Backdoor-adjusted diffusion and latent environments for intervention/counterfactual generation. | The stated SCM has static confounding roles; Appendix H.3 discusses feedback/time-varying assumptions. Reviewed, not executed here. This is not evidence it fails every sequential setting. |
| [Causal Diffusion Models for Counterfactual Outcome Distributions in Longitudinal Data](https://arxiv.org/html/2604.12992) | Longitudinal outcome generation conditioned on history and treatment plans, with causal assumptions. | Paper reviewed. No author implementation executed. Our Gaussian counterexample is not a reproduction or refutation of its reported results. The sequential adjustment actually implemented requires direct verification. |
| [ADMIT](https://arxiv.org/html/2609.34433) | Sequential interventional trajectory generation, admissibility constraints and adaptive policies. | Strong overlap with a generic causal trajectory framework. Reviewed, not executed. A new proposal must demonstrate something beyond its existing sequential factorization. |
| [TDS, NeurIPS 2023](https://arxiv.org/abs/2306.17775), [author repository](https://github.com/blt2114/twisted_diffusion_sampler/tree/5b6af53a6588c1bdd83d8604a8352fefdcad7d69) | Weighted diffusion sampling with asymptotic correction. | Only its native generic Feynman-Kac and resampling utilities ran. Original denoiser/twisting construction did not run. Three source files retained privately with checksums. Generic particle correction is already established. |
| [DiffPO, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/4d254c96c5ff500dda2ac0a58987aeba-Abstract-Conference.html) | Propensity-based orthogonal diffusion training for potential outcomes. | Existing training-time correction; different from frozen-generator reuse. Not executed. |
| [Diffusion Forcing](https://arxiv.org/abs/2407.01392) | Causal sequence sampling and guidance with flexible noise levels. | Earlier causal diffusion/guidance boundary. Not executed here. |
| [BridgeTwist](https://arxiv.org/abs/2610.18658) | Tempered twisted SMC in a Schrödinger bridge setting. | Recent abstract-level screen only; no full-paper sufficiency or failure claim. |

The candidate niche is restricted to reuse of a frozen learned whole-trajectory generator with limited query access, versus retraining an already causal model. That niche remains a question, not a novelty finding. A retained joint density, autoregressive factorization, and a conditional-only score oracle provide different capabilities and must not be conflated. Before method design, compare strongest existing sequential generation and generic posterior-sampling tools on the exact same nonlinear query. Current evidence is Gaussian and cannot establish a gap for real foundation models.

Previously closed controls remain relevant: C62 native PRO-DYN beat naive long rollout on ETTh1; C60 native NSSM repaired two measured history-state tasks; C61 ordinary HSMM repaired duration tasks; C63 audited a native COFE component but did not train it. Those findings rule out declaring generic rollout, memory, duration, or noise-normalization novelty from the current study. See their published artifacts for numerical evidence rather than aggregating incompatible tasks.
''')
save('baseline_registry.md',f'''# Baseline, data and reproduction registry

All C64 models share the generated trajectory information; no future held-out outcome or interventional truth selects a model or setting. Full joint Gaussian and Markov causal AR have different capacity. This is attribution, not a fair claim of new-model performance.

{table(['Entry','Execution and information','Status'],[
 ['Full joint Gaussian MLE','48 related prefix fits; 12 independent training corpora; covariance jitter 1e-10','Completed; all checkpoints hashed'],
 ['Causal linear Gaussian AR','48 train-only SVD fits, same records and prefix; policy regression only for observational NLL conditioning','Completed; existing causal baseline'],
 ['Known-policy density ratio','Privileged exact logging policy; 96 fixed queries','Completed; invalid precision retained'],
 ['Estimated-policy density ratio','48 train-only regressions; 96 fixed queries','Completed; invalid precision retained'],
 ['Fixed eigenvalue floor','1e-3 fixed before scoring, applied only estimated-policy precision','Completed; no tuning'],
 ['Frozen joint temporal factorization','Same full joint covariance; no new data/parameters/fit','96 rows completed; added after invalid-ratio screen'],
 ['Terminal importance sampling','Oracle Gaussian laws; 16 cells x 3 budgets x 3 MC seeds','144 rows; no learned score'],
 ['Annealed Gibbs SMC','32 temperatures, 1 sweep, native FK utility, 4 cells x 2 budgets x 3 seeds','24 rows; not full TDS'],
 ['Affine DDIM propagation','Observed / denoised-mean plug-in / exact target scores, 64/256/1024 steps','144 exact propagated Gaussian final-law rows'],
 ['Exact target iid','Privileged statistical floor, same population sizes, 3 MC seeds','144 post-screen reference rows; not deployable'],
 ['Real nonlinear generative framework','CaTSG/ADMIT/learned sequence diffusion or autoregressive density','Not trained or evaluated in C64']])}

Training corpus: each of 4 gains and 3 seeds has 16,384 length-64 training trajectories plus 4,096 independent validation trajectories. Horizons 1/4/16/64 are related prefixes. There are 48 predictive-law fit cells in each of two model classes and 48 policy regressions, not 96 independent datasets or paper reproductions. MLE uses no iterative optimizer.

Environment: Python {platform.python_version()}, NumPy {np.__version__}, SciPy {scipy.__version__}, Torch {torch.__version__}; CPU only. Generated data, 48 checkpoints and raw traces are stored on G in the authorized project subtree. No original source/data paths were deleted. The upstream optional numba JIT import/decorator is removed through AST for compatibility; original Python algorithm bodies remain intact. Native tilt/resampling/ESS fixtures qualified before execution.
''')
gg=[r for r in summary['learned_groups'] if r['horizon']==64];gr=[]
for k in [0.,.25,.75,2.]:
 for method in ['observational_block','causal_AR','ratio_known_policy','ratio_estimated_policy','fixed_eigenvalue_floor','same_frozen_joint_temporal_causal_factorization']:
  r=next(x for x in gg if x['gain']==k and x['method']==method);gr.append([k,method,f"{r['invalid']}/{r['rows']}",'undefined' if r['mean_discrepancy'] is None else f"{r['mean_discrepancy']:.6g}"])
nll=[]
for k in [0.,.25,.75,2.]:
 rr=[r for r in f['fits'] if r['gain']==k and r['horizon']==64];nll.append([k]+[f"{np.mean([r['validation_observed_conditional_NLL_per_step'][m] for r in rr]):.6g}" for m in ['oracle','block_Gaussian','causal_AR_reconditioned']])
smc=[[r['gain'],r['horizon'],r['particles'],f"{r['iid_mean_discrepancy']:.6g}",f"{r['SMC_mean_discrepancy']:.6g}",f"{r['iid_passed']}/3",f"{r['SMC_passed']}/3"] for r in i['groups']]
dd=[[r['DDIM_steps'],r['method'],f"{r['normalized_Gaussian_moment_discrepancy']:.6g}",f"{r['variance_trace_ratio']:.6g}",f"{r['terminal_tail_abs_error']:.6g}"] for r in s['DDIM_law_rows'] if r['gain']==2. and r['horizon']==64]
pop=[[r['gain'],r['horizon'],r['finite_second_moment'],f"{r['population_ESS_fraction']:.6g}"] for r in s['population_IS']]
save('experimental_results.md',f'''# Executed results

Scope: fully observed synthetic linear Gaussian feedback only. All prespecified learned-law rows, including invalid corrections, are in `outputs/FIT_AND_GAUSSIAN_EVALUATION64_V1.json`; all supplementary same-joint rows are in `outputs/CAUSAL_FACTORIZATION64_V1.json`. Sampler and iid-reference files retain every replication. No real-data or learned neural generator result is claimed.

System: Y_t=.8Y_(t-1)+.7U_t+.5epsilon_t; U_t=-kY_(t-1)+.5eta_t; independent Gaussian innovations; Y0=.25. Logging policies have full support, not a uniform propensity lower bound. Fixed queried plans are zero and sinusoid; queries are never used for fitting/selection.

At H64, the table averages six related evaluations (3 independent training seeds x 2 fixed query plans). Discrepancy is sqrt(W2_squared/trace(true covariance)) between Gaussian laws. Invalid rows are not silently averaged away. The remaining horizon tables are in `outputs/SUMMARY64_V1.json`.

{table(['gain','method','invalid/all','mean normalized discrepancy'],gr)}

H64 independent validation observational conditional NLL per step (3 training-seed mean). The causal AR is first reconditioned on the entire validation action plan, so these columns score the same observational conditional target. Low observed NLL does not itself certify intervention fidelity.

{table(['gain','oracle','block Gaussian','causal AR reconditioned'],nll)}

Across all horizons/queries, 16/192 policy-ratio laws are invalid (8 known-policy, 8 estimated-policy). Exact policy removal from a finite fitted Gaussian need not remain integrable. This does not invalidate the exact data-generating density-ratio identity. The fixed precision floor guarantees an algebraic Gaussian, but is not a faithful repair. Same-joint temporal factorization is always normalized and gives a strong no-retraining reference; its remaining finite-estimation error is larger than the correctly specified, much more parameter-efficient causal AR in the hardest cell.

Population terminal importance weights, sinusoidal query. A non-PD 2Q_do-Q_obs means infinite second moment, reported population ESS fraction zero; finite empirical ESS does not certify coverage.

{table(['gain','H','finite second moment','population ESS fraction'],pop)}

Finite-population references and ordinary SMC (means over three independent MC seeds). Metrics concern Gaussian projection of empirical moments, not exact empirical-law W2. The original operational rule was discrepancy<=.1 and terminal tail error<=.03 in each replicate. It was fixed before initial SMC execution; exact iid references were added after the screen and are disclosed as such. The rule itself rejects all H64/P512 exact iid replications. Therefore threshold failures alone cannot diagnose an algorithmic research gap.

{table(['gain','H','particles','iid discrepancy','SMC discrepancy','iid pass','SMC pass'],smc)}

SMC still has larger moment discrepancy than the iid reference in these fixed cells; no temperature/sweep/budget search was performed. That leaves efficiency room for this sampler configuration, but does not show failure of full TDS, stronger SMC or causal generators. Tail errors are retained separately in the raw JSON; a small tail error alone cannot establish the full law.

Oracle affine DDIM at gain2/H64. Its unchanged observed, plug-in-corrected and exact-target scores share discretization and initial N(0,I). The exact-target control separates solver error from wrong guidance.

{table(['steps','score','normalized discrepancy','variance trace ratio','terminal tail abs error'],dd)}

The denoised-mean plug-in correction does not equal the noisy log expected density-ratio score. Exact noisy-target scoring substantially repairs this oracle problem. Both facts are known posterior-guidance considerations; no new sampler follows from them alone. Score rows at all fixed noise levels and full 144 DDIM laws are retained.

Independent replay: PASS. 12 raw-corpus hashes, 48 checkpoint hashes, 480 original learned-law rows, 96 same-joint rows, 168 particle-law rows and 144 DDIM final-law moment rows checked. Alternate QR regression, direct sequential covariance regression, structural innovation covariance and SVD Bures arithmetic give maximum scaled discrepancy {max(a['max_scaled_errors'].values()):.6g}. Independent iid-reference arithmetic max absolute W2 difference {i['independent_metric_max_absolute_error']:.6g}. Oracle structural-conditioning difference {q['max_oracle_conditional_difference']:.6g}; noisy-ratio score finite-difference error {q['noisy_log_ratio_score_fd_error']:.6g}. This replay does not independently execute a full denoising pipeline.

Fit campaign elapsed {f['elapsed_seconds']:.3f}s; sampler campaign elapsed {s['runtime_seconds']:.3f}s, measured on CPU and excluding setup, reference and replay. These are execution receipts, not a matched-compute speed claim. Sampling seeds and shared prefixes are not independent biological systems.
''')
save('negative_evidence.md','''# Negative evidence and rejected claims

- Reject a new-framework claim from block conditioning bias: causal factorization and feedback identification are established, and the ordinary causal AR is strong here.
- Reject a claim that every frozen model requires retraining: the retained full joint checkpoint supports a valid temporal causal factorization without new fits or data.
- Reject treating the 16 nonintegrable ratios as failure of the exact causal identity. They concern removing one policy from an independently approximated observational density. Compatibility error matters.
- Reject the fixed eigenvalue floor as a solution: in the hardest H64 cell it yields normalized error about 37.3. No floor search or test tuning followed.
- Reject interpreting the SMC threshold failures alone as a gap: H64 iid exact-target references fail that absolute threshold too. Excess moment discrepancy remains in this particular low-budget sampler, with only three MC replications and no full TDS reproduction.
- Reject treating high empirical ESS as a coverage certificate when the population importance-weight second moment is infinite.
- Reject blaming the entire plug-in DDIM error on discretization: exact target scores at identical budgets yield much lower error. Conversely, exact target scores are privileged and are not a practical learned-model solution.
- Reject claiming CaTSG, ADMIT, DiffPO or the longitudinal causal-diffusion paper is empirically defeated: none was executed in C64.
- Reject framing an existing Gaussian information identity, inverse propensity adjustment, g-computation, or posterior SMC as a newly invented mechanism.
- Reject extrapolation to hidden confounding, partial observability, nonlinear learned models, transfer, clinical decision-making or real counterfactual truth. They were not tested.
- Reject counting the 48 horizon fits as 48 independent datasets; they are prefixes of 12 corpora. No GPU/neural training occurred.

Engineering exclusions: H1 has a fixed previous state, so rho/intercept and policy slope/intercept are not separately identifiable. v3 replay uses a fixed QR rank cutoff and checks predictions/noise rather than arbitrary equivalent coefficients. Earlier failed replay attempts changed no scientific setting and created no scored result.
''')
save('hypothesis_space.md','''# Competing hypotheses and tests

| Hypothesis | Discriminating test | Current disposition |
|---|---|---|
| H1: observed whole-plan conditionals already equal interventions under sequential ignorability | Exact structural joint law vs intervention product; gain0/H1 negative controls | False for feedback H>1; this is an existing causal distinction. |
| H2: poor observational fitting fully explains the failure | Oracle observed law and its intervention mismatch; correct validation NLL | False as sole explanation; oracle conditioning already differs. |
| H3: exact policy knowledge makes ratio correction of an estimated block law safe | Known-policy precision subtraction, compare estimated-policy and exact identities | False for fitted approximations in some fixed cells. Exact generative identity remains valid. |
| H4: every frozen model requires new training | Temporal factorization of the same retained joint covariance | False for this Gaussian joint checkpoint. |
| H5: low-cost denoised-mean correction equals exact noisy causal guidance | Analytic noisy scores and identical-budget affine DDIM | False in these oracle cells. No learned generator conclusion. |
| H6: an ordinary causal model already solves this attribution problem | Correctly specified causal AR with same train data | Supported in measured cells; main reason not to accept a new framework. |
| H7: SMC operational failures prove new sampling machinery is necessary | Exact-target iid statistical floor; stronger sampler budget/sweep controls if justified | Not established. Original H64 threshold also rejects exact iid samples. |
| H8: efficient reuse of a frozen nonlinear conditional-only generator remains practically harder than existing causal generation | Train-only nonlinear generator; strongest sequential/SMC controls; sealed mechanism/query tests | Untested. Only remaining provisional question, not a supported research gap. |

Next tests must define what access is retained: full joint density, autoregressive factors, score-only generator, policy observations and total compute. Causal AR and same-checkpoint temporal factorization are compulsory whenever mathematically available. A conditional-only access restriction must be motivated by an actual released model API, not invented to disable a strong baseline.
''')
save('research_ideas.md','''# Candidate disposition

Accepted research directions: 0. This is a negative feasibility update, not a paper proposal.

The only prioritized provisional test is whether an actual frozen nonlinear whole-trajectory generator can be reused for sequential interventions at a useful cost, beyond ordinary causal generation and existing posterior samplers. This is relevant to a reusable time-series world model across logging policies; no particular dataset is the objective. The novelty boundary is narrow and unresolved.

Do not propose "causal world model", "inverse policy correction", "causal masking", or "SMC-guided diffusion" as the idea: those labels cover established methods. Gaussian C64 provides a controlled attribution harness, not evidence of an unsolved nonlinear capability. A retained full joint Gaussian has an immediate known repair; correctly specified causal AR is stronger still.

A small nonlinear viability test should first reproduce an actual available sequence generator and ordinary sequential density baseline, then compare observation fidelity and intervention laws without selecting on test truth. Use different nonlinear mechanisms and logging policies with explicit sequential exchangeability, plus a separate held-out mechanism. Include exact-model/oracle diagnostics only as privileged references. Require calibrated distributional error relative to iid sampling floors and matched available information and compute. If existing controls already suffice, abandon this direction before developing a new architecture.

No second or third recommendation is invented to fill a list. Existing project evidence has already rejected several generic memory, duration, latent modularity and rollout narratives.
''')
save('final_research_direction.md','''# Research judgment after C64

**Research Question:** Can a real frozen nonlinear whole-trajectory time-series generator be reused for sequential interventions, with defensible distributional fidelity and useful inference cost, beyond existing causal factorization and posterior sampling?

**Evidence:** C64 proves an attribution phenomenon on a fully observed Gaussian feedback process. Direct block conditioning is wrong for intervention; independently estimated policy removal can become nonintegrable; a denoised-mean plug-in score remains inaccurate even with many DDIM steps. These are controlled facts, not a demonstrated novel gap. Causal AR largely repairs the problem; same-joint temporal factorization is a valid no-refit control. Original SMC threshold failures are partly confounded by the finite-particle sampling floor.

**Existing Methods:** Causal AR, temporal factorization and sequential g-computation already address causal semantics. CaTSG, ADMIT, DiffPO, Diffusion Forcing and TDS occupy nearby causal-generation/guidance territory. C64 has not shown their inadequacy on an actual learned nonlinear checkpoint. It would be unoriginal to rename these ingredients as a new framework.

**Proposed Idea:** No new mechanism is accepted. First test the narrow frozen-generator reuse question using an actual released architecture and access pattern. A proposed correction or distillation method should be designed only after an existing-method failure survives fair-information, training, sampler and iid-floor controls. Ordinary posterior expectation guidance, SMC and sequential density factorization are baselines, not innovations.

**Falsification:** Abandon if ordinary autoregressive/causal generation, available same-checkpoint factorization, or a strong generic conditional sampler reaches the target at comparable total cost. Abandon a score-only access limitation if it is not representative of an actual intended model. Also abandon if error is explained by low observational fit, unsupported actions, hidden confounding, finite sampling, or unfair compute. A new mechanism must improve sealed held-out mechanisms and policies across multiple seeds without future truth tuning.

**Feasibility:** The current exact Gaussian harness, saved raw corpora, learned covariance checkpoints and qualified native FK component are ready. A small nonlinear sequence model and causal density baseline fit a single RTX5090, but the current Gaussian study used CPU only. No performance or runtime for an unimplemented nonlinear model is asserted. Existing GPU/native instability noted in C62 should be qualified before a long run, without altering unrelated environments.

**Next Experiment:** Inspect the real CaTSG/ADMIT or a comparably available sequential generator's preprocessing, causal assumptions and conditioning interface; choose a genuine reusable checkpoint/access pattern. Before training, register a small nonlinear multi-mechanism feedback test with one held-out mechanism, correct observational NLL and interventional distribution metrics calibrated to exact iid floors. Compare causal training and strongest applicable frozen-model factorization/SMC. Reject the candidate if these already solve it. This is the highest priority falsification test, not a final recommended paper direction.

Final decision: overall goal ACTIVE / UNACHIEVED. Accepted final directions: **0**. C64 is closed and reproducible; novelty remains unproven.
''')
env={'utc':utc(),'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'scipy':scipy.__version__,'torch':torch.__version__,'compute':'CPU','cuda_used':False}
dump(OUT/'ENVIRONMENT64_V1.json',env)
closure={'utc':utc(),'status':'CLOSED','cycle':64,'overall_goal':'ACTIVE_UNACHIEVED','accepted_final_directions':0,'reports':reports,'original_learned_law_rows':480,'same_joint_rows':96,'particle_rows':168,'DDIM_rows':144,'iid_reference_rows':144,'invalid_ratio_rows':16,'unique_training_corpora':12,'related_horizon_fit_cells':48,'trained_neural_models':0,'sources_receipt_sha256':sha(OUT/'SOURCES64_V1.json'),'unlicensed_upstream_sources_redistributed':False,'artifacts':{name:sha(OUT/name) for name in ['FIT_AND_GAUSSIAN_EVALUATION64_V1.json','CAUSAL_FACTORIZATION64_V1.json','SAMPLER_EVALUATION64_V1.json','INDEPENDENT_REPLAY64_V1.json','IID_REFERENCE64_V1.json','SUMMARY64_V1.json','QUALIFICATION64_V1.json','ENVIRONMENT64_V1.json']},'post_screen_additions':['same-joint temporal causal factorization','exact iid finite-population reference'],'publication':'pending'}
dump(OUT/'CLOSURE64_V1.json',closure)
status='''# Current research status V64

Updated 18 October 2026. Overall autonomous research goal ACTIVE / UNACHIEVED. Accepted final directions: 0.

C64 closed a Gaussian attribution study of frozen observational trajectory laws under sequential intervention. It is not a new method, learned diffusion reproduction or real-data result. See [cycle64 reports](cycle64/README.md).

At gain2/H64, across three independent training seeds and two fixed intervention plans, normalized Gaussian discrepancy was 0.834823 for direct block conditioning, 0.008559 for ordinary causal AR, and 0.108579 for the same frozen joint checkpoint's temporal causal factorization (no refit). All six known-policy and all six estimated-policy ratio laws in that cell were nonintegrable; over the full registered grid there were 16 invalid ratio rows out of 192. A fixed eigenvalue floor gave 37.3287 error in that hardest cell. This is finite estimated-density incompatibility, not failure of the exact causal identity.

Oracle plug-in DDIM at 1024 steps still had 0.761772 discrepancy at gain2/H64; identical-budget exact target scores gave 0.002991. This isolates guidance approximation from solver error but does not validate a new sampler. Ordinary Gibbs/annealed SMC had excess moment discrepancy in the fixed budgets. Crucially, all H64/P512 exact iid target references also failed the original <=.1 moment threshold, so the raw failure count cannot establish an algorithmic research gap.

Independent replay passed: all 12 raw corpora and 48 checkpoints hashed; 480 original learned-law, 96 same-joint, 168 particle and 144 DDIM final-law rows verified. Max scaled arithmetic difference 4.263e-10. Another 144 exact iid references were checked with max absolute W2 arithmetic difference 2.274e-13. Shared horizon prefixes are dependent; no neural training occurred. Same-joint and iid controls were added after the initial screen and disclosed, without changing original results or tuning settings.

Research decision: do not turn known conditioning bias, inverse policy weighting, causal factorization or generic SMC into a claimed innovation. The sole prioritized provisional question is actual frozen nonlinear generator reuse under sequential interventions, compared with strongest existing causal/sequential generation and posterior sampling. It remains untested; ordinary causal AR already repairs the present system. A real model API/access pattern and nonlinear held-out mechanism test are required before any new-framework claim.

Earlier C60-C63 artifacts remain intact: native NSSM and HSMM supplied strong existing repairs, C62 provides a partial native PRO-DYN reproduction with recorded crashes, and C63 is a qualified untrained COFE component audit. No prior negatives are erased or counted as novel successes. Raw arrays/checkpoints remain under authorized server G storage; compact code/config/reports/receipts are uploaded to GitHub. Upstream TDS files are not redistributed because a root license was not located. No result files are synchronized locally.
'''
with (PROJECT/'CURRENT_RESEARCH_STATUS_V64.md').open('x',encoding='utf-8') as out:out.write(status)
print(json.dumps({'status':'CLOSED','reports':len(reports),'closure_sha256':sha(OUT/'CLOSURE64_V1.json'),'accepted_final_directions':0,'nll':nll,'invalid_cells':dict(Counter((str(r['gain'])+'_H'+str(r['horizon'])) for r in a['invalid_rows']))}))
