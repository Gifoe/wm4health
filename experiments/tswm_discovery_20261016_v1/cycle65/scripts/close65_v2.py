from common65 import *
import numpy as np,sys,torch,scipy
f=read(OUT/'FITS65_V1.json');e=read(OUT/'EVALUATION65_V1.json');r=read(OUT/'INDEPENDENT_REPLAY65_V1.json');s=read(OUT/'SUMMARY65_V1.json');risk=read(OUT/'ORACLE_DSM65_V1.json');native=read(OUT/'NATIVE_ORACLE_SAMPLER65_V1.json');assert r['status']=='PASS' and f['models']==18 and e['generated_laws']==240;assert read(OUT/'recovery_state65_v1.json')['status']=='TRAIN_EVALUATE_REPLAY_SUMMARY_TERMINAL_SUCCESS';source_check();reports={'literature_review.md':sha(ROOT/'literature_review.md')}
def save(name,text):
 with (ROOT/name).open('x',encoding='utf-8') as out:out.write(text.strip()+'\n')
 reports[name]=sha(ROOT/name)
def table(head,rows):return '| '+' | '.join(head)+' |\n|'+'|'.join(['---']*len(head))+'|\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in rows)
main=[];seed_rows=[]
for g in s['groups']:
 main.append([g['mechanism'],g['query'],g['method'],g['laws'],f"{g['scores']['do']['mean_normalized_sliced_W1']:.6g}",f"{g['scores']['obs']['mean_normalized_sliced_W1']:.6g}",'reference' if g['mean_runtime_seconds'] is None else f"{g['mean_runtime_seconds']:.5g}"])
 for target in ['do','obs']:
  for x in g['scores'][target]['seed_means']:seed_rows.append([g['mechanism'],g['query'],g['method'],target,x['training_seed'],f"{x['mean']:.6g}"])
vr=[]
for fit in f['fits']:
 for cp in fit['checkpoints']:
  if fit['model'].startswith('CSDI'):
   vals=cp['validation'];vr.append([fit['key'],fit['model'],cp['step'],f"{vals['block']:.7g}",f"{vals['prefix']:.7g}",cp['sha256']==fit['selected_checkpoint']['sha256']])
risk_rows=[]
for x in risk['rows']:risk_rows.append([x['key'],x['view'],f"{x['realized_Bayes_validation_DSM']:.7g}",f"{x['population_irreducible_DSM']:.7g}"])
oracle_rows=[[x['key'],x['operation'],f"{x['do']['normalized_Gaussian_W2']:.7g}",f"{x['obs']['normalized_Gaussian_W2']:.7g}",f"{x['do']['variance_trace_ratio']:.7g}"] for x in native['rows']]
save('README.md','''# C65: native learned conditional diffusion under sequential intervention

Execution, scoring and independent replay are closed. Overall research goal remains ACTIVE / UNACHIEVED; accepted final directions: 0. This is a transfer/mechanism experiment using unchanged native CSDI algorithms, not reproduction of its original benchmark scores and not a new proposed framework.

Read experimental_results.md for all comparisons, including each training seed; negative_evidence.md for excluded interpretations; literature_review.md for novelty boundaries; final_research_direction.md for the next discriminating audit. Eighteen formal models, six independent synthetic corpora, two fixed intervention plans, 240 generated laws and 480 target-score views are retained. The target views are paired measurements of the same laws, not independent experiments. Raw arrays, checkpoints and oracle traces remain in authorized G storage with hashes.

The original native worker suffered an access violation during checkpoint creation after12 completed models. A first recovery coordinator also crashed during torch import. Isolated recovery completed four more fits, then two attempts failed in Tensor pickle serialization for the next model. A numeric NPZ/JSON backend passed a302-tensor real model/optimizer/RNG roundtrip; the final two models use it, with the coverage model resumed from its exact4000-step checkpoint. All12 old successful fits and failed files remain. Root cause of native crashes is not proven. See RUNTIME_ISOLATION_NOTE65_V2.md and NPZ_RECOVERY_NOTE65_V1.md. EXECUTION_IDENTITY65_V2.json identifies the final coordinator, format-dispatch loader and actual evaluator/replay versions.

For a fresh reproduction, provision the recorded Python/PyTorch/NumPy/SciPy environment and configured storage; fetch and hash-check the five MIT-licensed native files under upstream/csdi. Use the qualified isolated entry with original protocols and declared source versions. The original first-stage scripts and failed attempts remain for provenance. Recovery scripts depend on existing artifacts and are not a fresh-run entry point. A fresh uninterrupted run uses the same training algorithm/configuration, then the fixed oracle/evaluation/replay scripts; write to fresh output/storage paths because exclusive-create guards protect evidence. No substitute architecture or fake dependency is used. The unused linear-attention import is omitted only in the explicitly disabled branch under isolation.
''')
save('baseline_registry.md',f'''# Model, data and execution registry

| Component | Definition | Interpretation |
|---|---|---|
| Native CSDI, random | Original CSDI_base, native random masks, full Transformer attention | Existing observational conditional generator; source algorithms unchanged. |
| Native CSDI, coverage | Same model and optimizer, .5 random/.25 block/.25 prefix training mask | Ordinary query-distribution control, not a proposed contribution. |
| Frozen block query | Reveal all planned U and no Y | Observational conditional; not automatically a do law. |
| Frozen prefix query | At each t reveal only generated Y_<t and U_<=t; keep current Y | Classical sequential g-computation through the identical frozen checkpoint. |
| Causal MDN | Conventional two-input width224/224 MLP, three Gaussian mixture components | Existing Markov transition-density baseline. |
| Causal AR | Train-only linear Gaussian regression | Correctly specified on linear system; misspecified nonlinear control. |
| Exact iid references |512 samples of do or exact observational conditional law | Privileged finite-sampling floor, no fitting. |
| Bayes DSM and native sampler references | Analytic linear Gaussian conditioning and original DDPM sampler | Privileged attribution, not learned methods. |

CSDI has51377 parameters; causal MDN has53097. Same records and fixed optimizer-update budgets do not imply identical FLOPs or information constraints. MDN has the correct Markov factorization and uses only Y transition targets; CSDI learns joint Y/U conditionals. Target-scalar exposures and measured per-fit runtimes are preserved in FITS65_V1.json. No claim of matched compute or superior new architecture is made.

Six independent corpora: linear/nonlinear mechanisms x seeds2021/2022/2023. Each contains16384 training,4096 validation and8192 reserved observational-test trajectories of length8, with explicit array channel0=Y, channel1=U. Known initial state=.25; behavior U=-.75 previousY+.5eta. Innovations are independent, all states observed, and the continuous Gaussian behavior policy has full support. No hidden confounding or clinical semantics is asserted.

Training: CSDI channels32/layers2/heads4/diffusion embedding32/time16/feature8; native50-step quadratic .0001-to-.5 schedule, Adam.001, batch64. MDN Adam.001, batch512 transitions. Fixed checkpoints2000/8000; common block/prefix validation DSM chooses CSDI, one-step validation NLL chooses MDN. Test truth never selects settings or models. Native source commit7f24a436f08d98853a6b43d4f7f04e5a65ecdf27 with MIT license is retained.

Final successful formal updates={f['optimizer_updates']}; original discarded model updates >=8000, plus >=4000 duplicated recovery updates and qualification fixtures listed separately. The original12 successful fits use the ordinary startup stack; the six recovery fits use isolated execution with the same PyTorch2.8.0+cu128. Final two fits use numeric NPZ/JSON checkpoints; stored model/optimizer/RNG values are unchanged by the format. All scored models and validation checks are replayed under the common isolated runtime. NumPy source-path difference and failed native execution are explicit reproducibility limitations, not silently omitted.
''')
save('experimental_results.md',f'''# Frozen-model law results

All models and queries follow PROTOCOL65_V1.md and EVALUATION_SPEC65_V1.md. The primary distance is the exact empirical one-dimensional W1 averaged over128 fixed random unit directions plus8 coordinate axes, divided by the reference's mean projected standard deviation. It is a finite sliced distance, not full multivariate W1/W2. Each generated law has512 samples; each exact do/observational reference has8192 samples. Rejection sampling has a valid unit envelope and its acceptance/proposal counts are retained. No threshold was chosen from these results.

Each learned-method row averages3 independent training seeds x3 MC replications. The nine law values are not nine independent model fits. Per-training-seed means appear below. An iid reference row has only3 MC replications. Scores against do and obs are two target views of the SAME generated law. Native block queries should first be judged against obs, prefix/causal rollouts against do; both columns expose the semantic difference.

{table(['mechanism','plan','method','laws','do distance','obs distance','mean generation seconds'],main)}

Generation timings exclude saving arrays and metric computation. Each native block batch uses50 denoiser calls; each eight-step prefix batch uses400. The same particle-batch size512 applies. Causal MDN uses8 small transition calls. These are observed operational costs, not an equal-compute new-method comparison.

Per-training-seed means (each averages3 MC replications):

{table(['mechanism','plan','method','target','training seed','mean distance'],seed_rows)}

Validation learning-curve control; both fixed checkpoints retained, selected flag determined only by the mean of block/prefix DSM:

{table(['corpus','native mask regime','updates','block DSM','prefix DSM','selected'],vr)}

Exact linear Bayes-noise validation references use the same normalization, fixed masks and native seeded noise. The realized and population quantities differ by finite validation noise. This is not a nonlinear oracle and a small DSM excess does not prove arbitrary-query calibration.

{table(['corpus','view','realized Bayes DSM','population irreducible DSM'],risk_rows)}

Native50-step sampler with an exact Gaussian conditional noise predictor, propagated analytically and checked using the unchanged impute method on32768 total native draws. The supplemental distance below is Gaussian W2 normalized by true covariance trace and must not be numerically equated to the primary finite sliced W1 above. Even exact denoising leaves about12% variance loss in prefix do simulation for this schedule. This is an existing fixed-reverse-variance/discrete-sampling issue, not evidence requiring a new world-model architecture.

{table(['corpus/plan/operator','operator','Gaussian error vs do','Gaussian error vs obs','variance trace vs do'],oracle_rows)}

Independent replay PASS: {json.dumps(r['counts'],sort_keys=True)}. Every raw law is hashed, every projected empirical W1 independently integrated using SciPy, and all36 validation-budget checkpoints and18 selections are checked. Maximum scaled numeric discrepancy={max(r['max_scaled_errors'].values()):.7g}; metric-specific tolerances and errors remain in INDEPENDENT_REPLAY65_V1.json. Native algorithms and complete configuration identities remain explicit.

Two synthetic systems with independently fitted checkpoints do not demonstrate a shared general-purpose world model, transfer to a new mechanism, a real intervention effect or clinical utility. No method was selected by these fixed query results. No new architecture is proposed or validated in this cycle.
''')
save('negative_evidence.md','''# Negative evidence and exclusions

- Whole-plan conditioning is an observational target under feedback; direct do-law failure is not itself a new research gap. Prefix sampling and ordinary causal density models are compulsory existing controls.
- A prefix failure cannot automatically be called causal-model incapacity: the model may approximate the conditional poorly, masks may be out of training distribution, finite DDPM variance may shrink the law, or empirical distance may reflect512-sample variability. C65 retains controls for each of these explanations.
- Reverse variance matters even for an exact linear denoiser. A claim that any remaining variance loss proves a novel latent-dynamics deficiency is rejected.
- Query-mask coverage is a standard training-distribution repair. Any gain from that control is not a new world-model mechanism.
- The near-parameter-matched MDN is an ordinary causal transition density, not a proposed model. A new-framework necessity claim must survive it and stronger applicable existing methods.
- OrthoGen already supplies generative recursive g-computation and orthogonal distributional treatment learning. Vid2World and Diffusion Forcing already causalize sequence generation. Generic use of these ingredients is not novel.
- CSDI's published imputation/forecasting benchmark scores were not reproduced here. The native architecture/loss/sampler were transferred with explicitly different dimensions and data. Do not label this an original-paper result.
- No CaTSG, ADMIT, OrthoGen, Pythia or pretrained NormWear-2 performance is measured by the CSDI table. Literature/code inspection is not empirical defeat.
- Native Windows faults are retained with exit codes and traces. Twelve successful fits were preserved; one8000-update cell's zero-byte checkpoint was not treated as valid. Its rerun and the recovery import failure are not hidden in successful update totals. Root cause is unproven.
- The earlier user-site attribution was inaccurate. Actual NumPy origin was an injected old baseline project's venv. V2 runtime provenance corrects that statement; isolation does not establish a causal hardware/software diagnosis.
- Results concern two fully observed known synthetic mechanisms. They do not establish robustness to hidden confounding, latent-state compression, adaptive policies, unseen systems, real medical interventions or individual shared-noise counterfactuals.
''')
save('hypothesis_space.md','''# Competing explanations after execution

| Hypothesis | Discriminating evidence retained | Allowed conclusion |
|---|---|---|
| Whole-plan conditioning suffices for intervention | Block law vs separate exact obs/do references | Compare targets explicitly; observational fidelity alone cannot identify a do law. |
| Frozen weights cannot support causal queries | Prefix and block operators using SAME frozen native CSDI checkpoint | A working prefix law falsifies a universal retraining requirement. The operation itself is existing g-computation. |
| Mask distribution is the principal problem | Random vs coverage training, fixed validation and test operators | Report all paired seed values. A coverage gain is a known repair, not architecture novelty. |
| Models were simply undertrained |2000/8000 validation trajectory and exact linear Bayes-risk reference | Residual learned approximation remains possible, especially nonlinear; avoid overattribution. |
| Sampler numerics explain law differences | Exact Gaussian denoiser through native50-step sampler | About12% prefix variance loss exists without any learned-model error in this linear configuration. |
| Empirical distance alone proves an algorithmic gap | Budget-matched exact iid do/obs sampling floor | Compare finite-sample variability; nonzero empirical distance is expected. |
| A new world-model framework is needed | Conventional causal MDN/AR plus current generative causal prior art | Not established by this study. No new method or independent-domain evidence exists. |

Tables in experimental_results.md and machine outputs supply numerical outcomes, including unfavorable cells. Mechanism-level explanations should follow these controls rather than treating the lowest aggregate distance as a research contribution. A framework proposal would still need a distinct unsolved capability, held-out mechanisms and appropriate real-data evidence.
''')
save('research_ideas.md','''# Candidate research judgment

Accepted final directions:0. Generic causalization of a conditional generator is not a new framework: conditional-generative causal composition, g-computation, diffusion forcing, guided sampling and orthogonal generative learners already exist. C65 adds measured feasibility/attribution using a real architecture; it does not overturn those novelty boundaries.

The remaining concrete question is what actual released pretrained time-series world models promise at the interface between conditioning, evolving state and controlled simulation. This is an audit target, not an accepted idea. The next available foundation-scale artifact is the pinned187M NormWear-2 checkpoint; Pythia still marks source/model release as pending in its primary report. Inspect actual inference and transition code before imposing causal/probabilistic claims or testing performance.

Do not rename a mask change, ordinary causal density head, variance correction, checkpoint bugfix or better sampler as the innovation. If a native pretrained framework fails only because of a software defect or an existing prefix/clamping/normalization repair, repair the baseline and reject that candidate. Only a persistent capability problem beyond existing methods could justify a new mechanism.
''')
save('final_research_direction.md','''# Current direction remains unaccepted

**Research Question:** Which reusable state/dynamics capability, beyond established conditional generation and causal factorization, is missing from actual time-series world-model frameworks? Current evidence does not yet identify a novel answer.

**Evidence:** C65 executes native learned CSDI under feedback, with random/covered masks, frozen block/prefix queries, ordinary causal MDN/AR, exact conditional truths, iid floors, Bayes-risk and native sampler references. All quantitative comparisons and failures are retained and independently replayed. It demonstrates measured operators on two systems, not a new method or cross-domain framework.

**Existing Methods:** Sequential g-computation, conditional-generative composition, diffusion causalization, ordinary transition densities and orthogonal generative learners are established. Correct reverse variance is also established. A future idea must exceed these controls rather than merely repeat them.

**Proposed Idea:** None accepted. The broad frozen-generator-to-causal-world-model narrative is insufficient as a novelty claim. Retain the attribution harness, stop proposing a new architecture from these ingredients alone, and inspect a real pretrained framework before selecting a different substantive failure.

**Falsification:** Reject a candidate if ordinary mask coverage, prefix queries, known-action clamping, consistent normalization, a correct sampler, or a conventional causal density solves it. Reject a reported failure if it comes from training quality, unsupported inputs, numerical/runtime faults or a task the model did not claim. A framework claim needs multiple independent mechanisms, sealed tests and an appropriate original-task baseline.

**Feasibility:** A single RTX5090 executed the declared CSDI/MDN studies; raw data/checkpoints remain on authorized G storage. The corrected isolated runtime and atomic snapshots preserve evidence after native failures. NormWear-2's public safetensors artifact fits the hardware, but staging alone is not successful execution or a reproduction result.

**Next Experiment:** Complete the source/weight/interface audit of pinned NormWear-2. First verify strict native weight loading and model I/O. Then examine observed/hidden channel roles and the stated planned-action interface using distinct semantic contracts; qualify ordinary repairs before any causal or performance conclusion. A synthetic API fixture is only an implementation audit. A research gap would require an independently selected controlled-system or appropriate physiological evaluation with real uncertainty/causal ground truth and strong existing baselines. If those repairs suffice, change research focus rather than optimize this benchmark.

Overall goal ACTIVE / UNACHIEVED. No final paper direction or top-conference novelty is claimed.
''')
env=dict(utc=utc(),python=sys.version,numpy=np.__version__,torch=torch.__version__,scipy=scipy.__version__,runtime='isolated original conda runtime with project injection excluded',algorithm_change=False)
dump(OUT/'ENVIRONMENT_FINAL65_V1.json',env)
artifacts=['FITS65_V1.json','EVALUATION65_V1.json','INDEPENDENT_REPLAY65_V1.json','SUMMARY65_V1.json','ORACLE65_V1.json','ORACLE_DSM65_V1.json','NATIVE_ORACLE_SAMPLER65_V1.json','RUNTIME_ISOLATION65_V1.json','MODULE_ORIGIN_CORRECTION65_V1.json','EXECUTION_IDENTITY65_V2.json','CHECKPOINT_IO_QUALIFICATION65_V1.json','NPZ_RECOVERY_SOURCE65_V1.json','ADJACENT_CODE_REVIEW65_V1.json','ENVIRONMENT_FINAL65_V1.json']
dump(OUT/'CLOSURE65_V1.json',dict(utc=utc(),status='CLOSED',reports=reports,artifacts={n:sha(OUT/n) for n in artifacts},unique_training_corpora=6,trained_models=18,successful_optimizer_updates=144000,original_discarded_updates_lower_bound=8000,additional_recovery_duplicated_updates_lower_bound=4000,generated_laws=240,scored_target_views=480,native_sampler_oracle_qualified_draws=32768,accepted_final_directions=0,overall_goal='ACTIVE_UNACHIEVED',publication='pending',runtime_caveat='Four native access violations across original/import/recovery stages.12 complete fits reused; four recovered with isolated PT snapshots, last two with qualified numeric NPZ/JSON. All common-runtime replay checked. Cause remains unproven.'))
status=f'''# Current research status V65

Overall goal ACTIVE / UNACHIEVED. Accepted final research directions:0.

C65 executed18 native-CSDI/ordinary-MDN fits on6 independently seeded corpora across two known feedback systems, retaining240 generated laws and480 paired obs/do score views. All36 validation-budget checkpoints and all law metrics passed independent replay. See [cycle65](cycle65/README.md) and its complete tables; this is a transferred architecture mechanism audit, not original CSDI benchmark reproduction or a new framework.

Block conditioning, frozen-prefix g-computation, query-mask coverage, causal density models, exact iid references and exact linear denoiser/sampler controls separate semantic, training, sampling and finite-population effects. The native50-step sampler itself loses about12% variance even with an exact linear denoiser. Generic causalization, masking or reverse-variance fixes are established methods; no novelty is claimed from them.

Runtime disclosure: original training suffered0xc0000005 during checkpoint serialization after12 complete fits, leaving an invalid zero-byte checkpoint. A recovery import also failed; two isolated recovery attempts later failed in Tensor pickle serialization. Isolation excludes an injected old baseline venv path, retaining PyTorch2.8.0+cu128 with intended NumPy2.2.6. Four fits recovered with CPU PT snapshots and separate validation; a numeric NPZ/JSON backend with302-tensor exact roundtrip recovered the final two, resuming one at4000. All12 old fits passed common-runtime replay. Root cause remains unknown. EXECUTION_IDENTITY65_V2.json records final source identities. Successful formal updates144000; discarded original>=8000 and duplicated recovery>=4000, fixtures separately logged.

The next source/weight audit targets an actual released pretrained framework, NormWear-2, with explicit distinction between observational completion and causal simulation. Staging or API fixtures do not constitute a performance result. Pythia's primary report still marks code/model to be released. Do not force an idea from existing causal-generation ingredients or software bugs; stronger independent-mechanism/original-task evidence is still missing.

All older C60-C64 artifacts and unrelated projects are preserved. Raw arrays/checkpoints remain on server G storage. GitHub receives compact code, reports and receipts; no result files are synchronized locally.
'''
with (PROJECT/'CURRENT_RESEARCH_STATUS_V65.md').open('x',encoding='utf-8') as out:out.write(status)
print(json.dumps(dict(status='CLOSED',reports=len(reports),closure_sha256=sha(OUT/'CLOSURE65_V1.json'),accepted_final_directions=0)))
