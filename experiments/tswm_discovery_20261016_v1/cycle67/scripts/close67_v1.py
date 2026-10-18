import sys,pathlib
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common67_v1 import *
a=read(OUT/'ANALYSIS67_V1.json');r=read(OUT/'INDEPENDENT_REPLAY67_V2.json');cm=read(OUT/'CASE_MANIFEST67_V1.json');sel=read(OUT/'VALIDATION_SELECTION67_V1.json');assert r['status']=='PASS' and read(OUT/'campaign67_v1_exit.json')['exit_code']==0 and read(OUT/'independent_replay67_v2_exit.json')['exit_code']==0
reports={}
def save(name,text):
 with (ROOT/name).open('x',encoding='utf-8') as f:f.write(text.strip()+'\n')
 reports[name]=sha(ROOT/name)
def table(head,rows):return '| '+' | '.join(head)+' |\n|'+'|'.join(['---']*len(head))+'|\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in rows)
def role(method):
 if method.startswith(('analog','ridge')):return 'known plan + TRAIN-domain labels'
 if method in ['native_predict','context_mean','last_value','seasonal_day']:return 'context only'
 if method=='bayes_ignored':return 'context transition; plan matching ignored'
 return 'known plan; frozen backbone/context fit'
tables={p:table(['method','information group','participant-mean normalized MAE','MAE mg/dL','mean case RMSE mg/dL'],[[g['method'],role(g['method']),f"{g['mean_normalized_MAE']:.8g}",f"{g['MAE_mg_dL']:.8g}",f"{g['RMSE_mg_dL']:.8g}"] for g in groups]) for p,groups in a['results'].items()}
contrasts=table(['subset','A minus B','normalized MAE delta','A better people /9','descriptive bootstrap95%'],[[x['split'],x['a']+' minus '+x['b'],f"{x['mean_paired_delta']:.8g}",x['a_better_participants'],str([round(v,8) for v in x['descriptive_participant_bootstrap_95percent']])] for x in a['contrasts']])
primary=[x for x in a['contrasts'] if x['a']=='bayes_action_full_view' and x['b']=='bayes_indexed'];evalc=next(x for x in primary if x['split']=='EVALUATION');largest=max(abs(p['delta']) for p in evalc['participant_deltas'])/sum(abs(p['delta']) for p in evalc['participant_deltas'])
people=table(['participant','full-view normalized MAE','indexed normalized MAE','difference'],[[x['participant'],f"{x['a']:.9g}",f"{x['b']:.9g}",f"{x['delta']:.9g}"] for x in evalc['participant_deltas']])
save('README.md','''# C67: pretrained conditional transition comparison on real numerical records

Real data and model execution are closed with independent replay. Overall research goal remains ACTIVE / UNACHIEVED; accepted final directions0. This is a frozen mechanism comparison in a supported physiological conditional-forecasting task family, not exact reproduction of the NormWear-2 paper tables, a new method or clinical causal validation.

45 public CGMacros participant numerical CSVs were acquired without food photographs. Official archive member lengths/CRC32 were checked on server, and each retained file's SHA256 recorded. The complete archive was not acquired, so its published whole-file SHA256 is not claimed verified. Source: https://physionet.org/content/cgmacros/1.0.0/ , DOI10.13026/3z8q-x658, CC BY-NC-SA4.0. Raw records, forecasts and pretrained weights remain server G; this snapshot contains authored code, reports and compact provenance only.

The one full pretrained186714897-parameter native model is reused across operators. Three ordinary ridge configurations are fitted on422 TRAIN windows from26 usable participants; analog retrieval has three predeclared k settings.9 validation participants/35 cases select k16 and alpha100 before9 EVAL participants/33 cases are scored.910 VAL plus726 EVAL prediction records are retained. Three MC seeds are sampling replications, not independent neural fits.

Main comparison: same-view action-centroid control versus indexed native transition. EVAL normalized MAE0.99167263 vs0.99813962 (0.648% lower), with9/9 participant means lower; VAL0.93325698 vs0.93281082 (0.048% worse), with4/9 lower. The EVAL improvement is real in the frozen measurements, but small and concentrated, and the repair has direct prior art. It does not establish a central unresolved world-model gap. Read experimental_results.md for every method/participant contrast, information differences and limitations. Do not continue tuning this dataset to manufacture a contribution.
''')
save('literature_review.md','''# Literature and novelty boundary

Read [the source-verified intake](LITERATURE_DELTA67_V1.md) for primary links and scope. ALER-TI explicitly studies incomplete-query/complete-candidate representation mismatch, including on-the-fly masking and latent alignment. AdaJEPA and Sandwich-Residuals already adapt pretrained world models using self-supervised transitions or surrounding residuals. Generic alignment, re-encoding or frozen-core adaptation is therefore insufficient novelty.

NormWear-2 is actually executed here at the pinned model revision. Its demonstration and reported task lengths informed a separately specified protocol; unavailable full evaluation pipelines/splits prevent a paper-table reproduction claim. No ALER-TI, AdaJEPA or Sandwich-Residuals score is measured by this experiment. Their applicability is a prior-art constraint, not an empirical defeat claim.

The user's target is a reusable time-series world-model capability. Familiar components do not automatically invalidate an effective framework, and ordinary success on a few low-dimensional systems does not settle every learning problem. The required increment is a specific useful capability, task-valid failure evidence, a distinction from applicable methods and independent-mechanism validation. C67 supplies a real-task negative/prerequisite result; it does not meet that complete increment.
''')
save('baseline_registry.md',f'''# Model/data/protocol registry

Pinned full native NormWear-2 revision3d1281c9994140056edfbbdedab57f2e6493dfaa; weight SHA25644ddf4118b4b027a5b726ce7368439e0fa28fd3c331909149befac2087daad11.186714897 parameters plus327680 buffers,277 strict native state tensors. Source algorithms and all weights are retained; C66's ordinary hidden-channel recurrent clamp and action-coordinate correction are explicit controls. Full/past-view and raw-covariate variants change only distance features under the same context clusters/transition counts/native sampler. They are not new trained architectures.

Three ridge coefficient matrices, each133x720, use TRAIN labels. Analog forecasting retrieves only TRAIN windows. Their target-domain information differs from zero-shot/context-only native adaptation, so the result table identifies this group. No equal-compute or pretraining-data comparison is asserted. The pretrained corpus's overlap with these released records is unverified; participant splits protect this experiment's fitting/selection, not unknown upstream pretraining.

Source45 participant files/687580 released one-minute rows. Split27/9/9 by seed67021; after fixed coverage rules,26 TRAIN participants provide422 bank windows, VALIDATION9/35 cases, EVALUATION9/33 cases. Channel roles are explicit[Libre GL,Carbs,Protein,Fat].2880-minute observed context,720-minute known-meal forecast,16 synthetic padding points use repeated last known macros and zero hidden response. Context normalization uses observed data only; source missing targets stay masked. Float32 data inputs/retained targets and native inference; classical feature/statistical calculations use float64 except source macro-block sums performed on float32 inputs.

Native Bayesian states10,4 fused sampled trajectories,3 fixed MC streams2021/2022/2023. Padded output is cropped explicitly. Native predict lacks future meals and is an information ablation. Point forecasts and native weighted trajectory fusion do not certify a calibrated predictive law.

Runtime: existing isolated Python3.12.13/PyTorch2.11.0+cu128/NumPy2.4.3 on RTX5090, TF32 disabled, no environment installation. pandas was absent; a failed authored audit import is retained, and final preprocessing uses stdlib CSV/ISO datetime plus NumPy. Successful source/precision corrections preserve every prior failure and do not rerun or alter frozen forecasts. See PROTOCOL67_V1.md, EXECUTION67_V1.json, source/data receipts and independent replay.

Selected classical settings: k={sel['analog_k']}, alpha={sel['ridge_alpha']}; selection SHA256{sha(OUT/'VALIDATION_SELECTION67_V1.json')}, written before EVAL. All predeclared native controls are reported, including unfavorable ones. No EVAL tuning occurred.
''')
save('experimental_results.md',f'''# Real-task results and independent evidence

Primary metric is per-case Libre MAE divided by that case's observed-context sample std; average MC/cases within each participant, then average participants equally. Raw MAE and case RMSE are in mg/dL; RMSE is averaged by case/participant, not a pooled global square root. Each subset has9 participants. MC streams are repeated samples from the same frozen model, not independent fits. Source resampling provenance, different future information and target-domain training must be respected.

Validation35 cases,910 prediction records:

{tables['VALIDATION']}

Classical k16/alpha100 were selected and hashed before EVAL. Evaluation33 cases,726 records:

{tables['EVALUATION']}

Main effect is mixed across subsets: full-view action re-encoding is0.048% worse on VAL and0.648% lower on EVAL than indexed native matching. All9 EVAL participant means improve, and the descriptive bootstrap interval is below0; do not call the effect absent or statistically zero. About{100*largest:.2f}% of the EVAL aggregate absolute decrease comes from one participant. The small effect and prior-art overlap do not justify promoting this repair into a new framework or treating view mismatch as the dominant error source. Native clamped simulation is already a stronger context-only-fit control here. Analog/ridge use additional TRAIN-domain labels and must be grouped separately.

Every prelisted descriptive contrast,10,000 participant-bootstrap draws, seed67025; these are exploratory unadjusted intervals, not nine times the number of windows or a corrected family of confirmatory tests:

{contrasts}

EVAL main contrast by participant:

{people}

Full-view versus past-view action controls differ by at most{a['full_vs_past_view_max_abs_forecast_change_mg_dL']:.9g}mg/dL over all retained forecasts. This establishes near-equivalence of measured outputs here, not a theorem about every internal representation or sequence length. Raw-macro matching improves VAL but does not preserve the improvement on EVAL. Conditioning's effect also depends on the reported metric; do not infer causal validity from an error difference.

Independent replay PASS: {json.dumps(r['counts'],sort_keys=True)}. All490 retained cases were reconstructed from raw CSVs; all68 forecast archives and1636 metric rows were hashed/recomputed;480 classical forecasts replayed; three ridge matrices verified by an independent centered SVD solution; two fixed native EVAL calls matched bitwise. Maximum raw-MAE discrepancy={r['maximum_absolute_errors']['MAE_mg_dL']:.4g}, ridge-coefficient discrepancy={r['maximum_absolute_errors']['ridge_SVD_coefficients']:.4g}. The first verifier stopped on a7.15e-7 float32 macro-sum versus float64-reference discrepancy. V2 kept strict history/metric tolerances and checked each macro feature against its explicit60*eps32*sum_abs arithmetic bound; no primary data/prediction/score changed.

Timings/call counts remain per row. The serial bundle caches conditioning views for ordinary controls, so these are not standalone equal-compute latency benchmarks. Raw numerical records and forecast arrays remain on G, with FORECAST_ARCHIVE_LEDGER67_V1.json listing all hashes. No physical counterfactual truth, uncertainty calibration, original-paper composite score or original-paper split reproduction was established. One independently specified dataset protocol is not a general framework validation.
''')
save('negative_evidence.md','''# Negative evidence and invalid inferences

- Same-view re-encoding produces a small positive EVAL effect, not a dominant repair: VAL reverses sign and the aggregate EVAL decrease is concentrated in one person. Preserve9/9 EVAL wins and the negative descriptive interval; do not suppress a measured positive just because it does not establish novelty.
- Channel-and-length-consistent versus channel-only action views yield nearly identical forecasts. A sequence-length mismatch explanation does not account for substantial error here.
- Raw matching's VAL advantage does not generalize to the frozen EVAL cohort. Further test-set tuning is prohibited.
- Ordinary aligned re-encoding already has close time-series retrieval prior art. A small improvement on this physiological task does not supply a method distinction or establish a new world-model framework.
- Native conditional transition is not clearly superior to its action-ignored ablation by the primary metric. That does not prove that nutrition is causally ineffective: information, estimator quality, sampling, normalization and cohort weights all differ from a causal estimand.
- Context mean and clamped native simulation are strong controls here. Analog/ridge have additional target-domain labels; their performance cannot support an equal-data foundation-model superiority claim.
- The held-out participant protocol does not certify disjointness from unavailable upstream pretraining data. No stronger zero-shot generalization is asserted.
- The released data provide one-minute trajectories, whereas the original sensors sample at15/5 minutes. Exact resampling/real-time availability is unverified, so strict physical/online causality cannot be inferred from these series.
- Sparse meal covariates are retrospective conditional information. Actual future meal logs are not an operational prediction of unannounced behavior or clinical intervention truth.
- The original paper used different undisclosed preprocessing/splits and multidimensional metrics; this point-error protocol cannot refute its published composite ranking.
- Source acquisition avoided photographs and verified member CRC/length plus server SHA records. It did not verify the complete archive SHA256. Native execution and independent replay do not remove this provenance boundary.
- The original float32 verifier discrepancy was arithmetic precision, not a changed performance result. Its trace and versioned correction are retained. No environment packages or frozen predictions were modified to obtain PASS.
''')
save('hypothesis_space.md','''# Competing hypotheses after the fixed real task

| Hypothesis | Evidence | Current interpretation |
|---|---|---|
| Wrong action-coordinate indexing is the whole problem | C66 indexing correction, retained as C67 baseline | Correct the software first; no architecture novelty follows. |
| Conditional-view mismatch dominates remaining forecast error | Indexed versus same-view transition, all other native components fixed | EVAL shows0.648% lower error, VAL slightly worse. This is not sufficient evidence of a dominant unresolved bottleneck. |
| Historical versus full-plan action-encoding length explains the effect | Past/full-view forecasts differ by only4.58e-5mg/dL max | No substantial measured output effect in these cases. |
| A simpler action distance would solve the problem | Raw-macro control | VAL improvement does not persist on EVAL. Do not tune EVAL. |
| More elaborate context transition is always better | Action-ignored, indexed, native predict/simulate and context-mean controls | Not supported by the primary point metric. This does not settle distributional dynamics or original composite metrics. |
| Ordinary supervised task reuse already gives a strong comparator | TRAIN-only analog/ridge selected on VAL | Yes in this protocol, with additional target-domain labels; not an equal-information causal comparison. |
| A new general framework has now been validated | One supported-task-family evaluation, no new method | No. Independent mechanisms, a distinct method and task-valid remaining failure are still absent. |

Separate a useful practical repair from a new research question. A reusable framework need not beat every classical model on every small system, but it must offer a specific useful capability beyond these existing components with a fair evidence package.
''')
save('research_ideas.md','''# Current research triage

Accepted final directions0. Lower the priority of query-view alignment as the main new research gap. Its ordinary repair has prior art and only a small inconsistent cross-cohort primary contrast here. Do not continue optimizing CGMacros or add architecture complexity to enlarge this particular effect.

Retain the actual full pretrained model, corrected native operators, raw-source verification and per-participant evaluation harness. These are evidence assets, not the proposed contribution. A new direction must identify a reusable state/dynamics capability, not a dataset score.

The next unresolved diagnostic is why a context-fitted latent transition operator adds little point-prediction utility beyond a stronger native/direct baseline. Possible causes include clustering/transition estimation, insufficient context, feature time/observation dependence, irreducible future uncertainty or an objective/task mismatch. These explanations must be separated on systems with valid ground truth before proposing a new representation or adaptation rule. Established DMD/Koopman, conditional regressions and state-space filters are compulsory controls; their components are not inventions.

This is a next diagnostic target, not an accepted idea or a claim that those methods fail. Prior work on Meta-iLaD, COFE, RP-GSSM, OSSM and online world-model adaptation must constrain any future claim. Read the actual methods/assumptions and use real implementations; do not infer a gap from missing release artifacts or impose a causal task on an observational model.
''')
save('final_research_direction.md','''# No accepted framework direction

**Research Question:** Which reusable state/dynamics capability remains unresolved after competent source implementations and standard context adaptation? C67 does not yet give a novel answer.

**Evidence:** Full pretrained NormWear-2 executed on45 public numerical records with frozen participant splits,68 VAL/EVAL cases and1636 retained point forecasts. Independent replay passed source reconstruction, all score rows, ordinary baseline fits and fresh native calls. Same-view action matching produces a small EVAL benefit but no VAL benefit; existing simple/model-native controls remain strong.

**Existing Methods:** Query/candidate alignment, on-the-fly re-encoding, frozen-core adapters, ordinary conditional analog/ridge estimation, latent state filtering and Koopman/DMD operators already exist. C67's small repair is not a distinct method. Classical controls on limited tasks do not prove that every reusable neural framework is unnecessary.

**Proposed Idea:** None accepted. Do not promote the alignment control or a source bugfix into a paper direction. The research-discovery objective remains unfinished.

**Falsification:** Reject the current dominant-view-mismatch story if simple re-encoding adds only small/inconsistent value, as observed here. For a subsequent operator/state hypothesis, reject a new-architecture necessity claim if a competent existing transition estimator with the same information removes the residual. Reject task-invalid or pretraining-overlap claims and test-set hyperparameter changes.

**Feasibility:** One RTX5090 runs the exact187042577-state-element native model and all fixed operators. Raw data, forecasts and weights stay on G. Only authored code/reports/provenance go to GitHub; no local result files. The current study is one independently specified physiological task, not a cross-domain new-framework validation.

**Next Experiment:** Stop this dataset/geometry tuning. On independently selected controlled dynamical systems with known future laws, hold the pretrained representation and available history fixed, compare its context-cluster transition to competent regularized conditional linear/DMD and state-space controls, plus direct native inference. Separate operator estimation from representation insufficiency and irreducible uncertainty. First audit existing source/protocols and current Meta-iLaD/COFE/RP-GSSM/OSSM evidence to avoid repeating closed experiments. A repaired ordinary operator ends that explanatory route; only a persistent task-valid capability gap could justify method design. This follow-up is not yet executed or accepted as a new research problem.

Overall goal ACTIVE / UNACHIEVED. Accepted final directions0.
''')
artifacts=['DATA_SOURCE67_V4.json','DATA_AUDIT67_V4.json','PARTICIPANT_SPLIT67_V1.json','CASE_MANIFEST67_V1.json','EXECUTION67_V1.json','NATIVE_QUALIFICATION67_V1.json','BASELINE_FIT67_V1.json','VALIDATION_RESULTS67_V1.json','VALIDATION_SELECTION67_V1.json','EVALUATION_RESULTS67_V1.json','INDEPENDENT_REPLAY67_V2.json','ANALYSIS67_V1.json','FORECAST_ARCHIVE_LEDGER67_V1.json','campaign67_v1_exit.json','independent_replay67_v2_exit.json']
dump(OUT/'CLOSURE67_V1.json',dict(utc=utc(),status='CLOSED_REAL_SUPPORTED_TASK_FAMILY_MECHANISM_EVALUATION',reports=reports,artifacts={n:sha(OUT/n) for n in artifacts},public_participant_files=45,training_windows=422,validation_cases=35,evaluation_cases=33,validation_participants=9,evaluation_participants=9,forecast_records=1636,neural_models_reused=1,new_neural_training=0,ordinary_ridge_fits=3,accepted_final_directions=0,overall_goal='ACTIVE_UNACHIEVED',raw_artifacts='server_G_only',publication='pending'))
status='''# Current research status V67

Overall goal ACTIVE / UNACHIEVED. Accepted final directions0.

C67 advances from C66's interface fixture to a frozen real supported-task-family evaluation.45 public CGMacros numerical files/687580 rows; TRAIN422 windows from26 usable participants, VALIDATION35 cases/9 people, EVALUATION33 cases/9 people. The complete native pretrained NormWear-2 model is reused; ordinary ridge/analog settings are chosen on VAL only.1636 forecast records and68 raw forecast archives are retained on G.

The main same-view action-matching control changes normalized MAE from0.93281082 to0.93325698 on VAL and0.99813962 to0.99167263 on EVAL. All9 EVAL participant means improve slightly, but the mean reduction is0.648%, concentrated in one participant, and the repair has direct alignment prior art. Past/full action-view forecasts differ by at most4.58e-5mg/dL. Lower this as a main research direction; do not tune this dataset further to manufacture a contribution. Complete results, contrasts, information groups and limitations are in cycle67/experimental_results.md.

Independent replay PASS:45 raw source hashes,490 reconstructed data cases,1636 score rows,480 classical forecasts, independent SVD ridge checks and two fresh native EVAL forecasts (bitwise agreement). A float32 feature-aggregation verifier correction is explicit and does not change results. Member lengths/CRC were checked and SHA256 recorded; the full archive checksum was not verified. Original-paper evaluator/splits and upstream pretraining overlap remain unknown; this is not a headline reproduction or clinical causal validation.

The next residual to audit is operator estimation versus reusable-state quality on independently selected systems with valid law references and strong existing DMD/conditional/state-space controls. No new framework is proposed or accepted. All C65/C66 and earlier artifacts/unrelated projects are preserved. Raw arrays/weights remain server G; GitHub receives compact evidence, with no result-file synchronization locally.
'''
with (PROJECT/'CURRENT_RESEARCH_STATUS_V67.md').open('x',encoding='utf-8') as f:f.write(status)
print(json.dumps(dict(status='CLOSED',reports=len(reports),closure_sha256=sha(OUT/'CLOSURE67_V1.json'),accepted_final_directions=0)))
