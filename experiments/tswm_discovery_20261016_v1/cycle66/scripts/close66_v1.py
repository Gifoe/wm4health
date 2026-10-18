"""Close an implementation audit; never convert its fixtures into performance claims."""
import pathlib,json,hashlib,datetime,math,os
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs';PROJECT=ROOT.parent
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def dump(p,obj):
 with p.open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,allow_nan=False)
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
q=read(OUT/'INTERFACE66_V2.json');b=read(OUT/'BAYES_INTERFACE66_V1.json');r=read(OUT/'INDEPENDENT_REPLAY66_V1.json');m=read(OUT/'MODEL_FILES66_V1.json');assert r['status']=='PASS' and read(OUT/'interface_exit66_v2.json')['exit_code']==read(OUT/'bayes_exit66_v1.json')['exit_code']==read(OUT/'replay_exit66_v1.json')['exit_code']==0
assert sha(q['array_artifact'])==q['array_sha256'] and sha(b['arrays'])==b['arrays_sha256']
for f in m['files']:assert sha(f['server_path'])==f['sha256']
reports={}
def save(name,text):
 with (ROOT/name).open('x',encoding='utf-8') as f:f.write(text.strip()+'\n')
 reports[name]=sha(ROOT/name)
def table(headers,rows):return '| '+' | '.join(headers)+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in rows)
rows=[[x['name'],str(x['shape']),f"{x['known_future_output_max_abs_change']:.8g}",f"{x['second_encoder_known_first_future_block_max_abs_change']:.8g}"] for x in q['simulate_runs']]
save('README.md','''# C66: released pretrained NormWear-2 interface audit

The real pretrained framework is loaded and executed, with native predict, simulate and latent Bayesian transition branches. This is an implementation audit, not an original-paper benchmark reproduction, held-out task result, new method or accepted research gap. Overall goal remains ACTIVE / UNACHIEVED; accepted final directions0.

Pinned model revision3d1281c9994140056edfbbdedab57f2e6493dfaa, full safetensors SHA25644ddf4118b4b027a5b726ce7368439e0fa28fd3c331909149befac2087daad11. Actual model parameters186714897 plus327680 registered buffers =187042577 state elements. The earlier protocol/staging field calling187042577 parameters is corrected by the executed receipt and BAYES_INTERFACE_PROTOCOL66_V1.md. The original wrapper-prefix assumption was also wrong: V1 stopped at an authored assertion before inference; V2 loaded all277 bare-key native tensors strictly and checked every loaded value bitwise. No weights were dropped and no proxy network was used.

Two ordinary implementation discrepancies were demonstrated: recurrent simulation overwrites known covariates, and the Bayesian action-centroid suffix selects mixed coordinates from a feature-major interleaved representation. Minimal clamping/indexing controls were executed and labelled. These repairs are not research innovations. Independent array/layout replay and fresh native CPU/GPU prediction/encoding checks passed.

Read experimental_results.md for exact quantities and limitations; negative_evidence.md excludes unsupported causal/performance claims. No upstream model source, weights or signal arrays are published here; pinned links and hashes identify them on server G. Raw source-derived diff remains server-only. Authored scripts and compact provenance are published.
''')
save('literature_review.md','''# Current literature boundaries

[NormWear-2](https://arxiv.org/html/2605.15465), May14,2026, supplies a pretrained multivariate backbone and a context-fitted latent transition branch. Its reported simulation task completes hidden channels given observed channels; its physiological action analysis is qualitative, rather than universal identification of interventional effects. This audit executes the released implementation at [pinned HF revision](https://huggingface.co/mosaic-laboratory/normwear2/tree/3d1281c9994140056edfbbdedab57f2e6493dfaa), without claiming reproduction of its dataset tables. Whole-plan conditional dependence and a nonanticipating controlled process are different contracts. Interface fixtures cannot adjudicate the paper's clinical performance.

[Rethinking World Models for Biosignals](https://arxiv.org/html/2610.17510), Oct13,2026, already distinguishes genuine actions and evidence of causal effects, and discusses temporal latency, system openness and endogenous dynamics. Those broad criticisms and an intervention-validation checklist are existing literature, not a new gap discovered here. We did not execute a method from this perspective article.

[ThermoForce](https://arxiv.org/html/2607.03942), July4,2026, already combines a frozen passive forecasting prior with a compact physics-structured forced-response operator identified using control excitation. Generic passive/forced separation or an action-response adapter cannot be claimed as new. Its task-specific assumptions do not automatically cover every nonlinear system; no empirical superiority or failure was measured here.

[Learning Transferable Policies from Action-free Time Series Through Dynamical Embeddings](https://arxiv.org/html/2610.03065), Oct2,2026, explicitly supplies an intervention model rather than identifying actuation from action-free recordings, and reuses shared dynamical embeddings in controllers. It is relevant prior art for embedding-based adaptation and control, not a model executed in C66.

[Pythia](https://arxiv.org/html/2610.05240) investigates shared multimodal predictive pretraining with a separate probabilistic readout. The previously inspected primary report marks code/model release pending. Release availability is not scientific novelty. C65 already records closer conditional-generator/g-computation and diffusion-causalization prior art. Neither implementation defects nor a missing release can justify a new framework claim.
''')
save('baseline_registry.md',f'''# Execution and baseline registry

Native full architecture/configuration is unchanged. The thin Hugging Face wrapper was bypassed by instantiating its exact delegated class under a package namespace for relative imports. No transformers installation or dependency imitation was introduced. All277 exact bare-key state tensors loaded strictly, with0 missing/unexpected keys and bitwise loaded-value checks. Parameters={q['parameters']}; buffers={q['buffers']}; state elements={q['state_elements']}.

Executed paths: native predict; native simulate; native unconditional and conditional Bayesian transition; explicitly labelled covariate-clamped simulate and action-coordinate-indexed Bayesian sampler. These are interface paths and existing-method repairs, not independently trained baselines. No task leaderboard, matched-compute gain or intervention-effect score is claimed.

Existing environment used without modification: Python3.12.13, PyTorch2.11.0+cu128, NumPy2.4.3, sklearn1.8.0, safetensors0.7.0, isolated execution, RTX5090, float32, TF32 disabled. This differs from C65's PyTorch2.8 runtime and must not be silently merged into a homogeneous reproduction claim. Source/module origins are retained in INTERFACE66_V2.json.

The fixture has explicit channel0=response/channel1=known covariate,128 observed and32 future points, fixed analytic nonconstant signals and seed2021. Native insight uses a16-point returned horizon plus one extra latent patch,3 context clusters,4 sampled latent trajectories. All normalization in that branch uses observed128-point statistics. This fixture has no physical transition truth and is not physiological data.

Source/weight metadata and transport receipts preserve an initially mismatching file. Direct server downloads failed; public content was streamed through transient controller RAM, with no local model/result file. Stable independent remote hashes exposed the mismatch. An ACKed selective repair retained the bad original, patched9 four-MiB chunks, and independently verified the full public SHA256. Cause of the original corruption is unknown. A first repair verification hit a Windows open-file sharing lock; its candidate was retained and re-inspected before continuation. Never load the mismatching original.
''')
save('experimental_results.md',f'''# Executed interface evidence

Full weight loading and native computation succeeded. Native predict returned[1,32,2] finite values. For the160-point simulate input, native output length is176 due to the current padding expression; full outputs are retained and diagnostics use the declared first160 points. The API mutates caller input by clearing hidden future response, so each run receives a fresh clone. These are implementation facts, not predictive-quality results.

{table(['path','output shape','known-channel output max change','second-call known-channel input max change'],rows)}

The native stored-tensor update changes all channels, including the known covariate, before the next encoder call. Two AST subscript changes restricting that update to hidden channels reduce the next encoder's known-covariate discrepancy from0.2809356153 to0 and output discrepancy from0.1281011403 to5.96e-8. Original source bytes remain unchanged. This is ordinary clamping, and no response accuracy improvement is inferred without ground truth.

Changing only hidden future responses by+111 produces exactly0 output change. Perturbing only the final16 known covariate points changes the first16 generated response points by{q['late_covariate_perturbation_earlier_response_max_abs_change']:.9g}. This is a whole-plan conditional-interface diagnostic. It is not a failure against a universal causal guarantee; conditional completion can depend on the whole known plan.

Native insight encoded observed history as[2,8,768], then flattened to[8,1536] by feature-major channel interleaving. The last768 coordinates therefore contain384 nonaction coordinates and omit384 action coordinates. Sentinel arithmetic and actual pretrained embeddings both verify that action channel1 occupies coordinates1,3,...,1535. Original unconditional/conditional and one indexing-only conditional repair each returned finite[1,16] outputs, with identical fitted3x3 transition matrices and exactly normalized rows. Indexing-only repair changes the returned conditional fixture by{b['indexed_vs_native_conditional_output_max_abs_change']:.9g}; this is not an accuracy gain. It also does not prove alignment between separately encoded actions and jointly encoded state/action centroids.

Independent replay PASS. Scope: {json.dumps(r['counts'],sort_keys=True)}. Fresh CPU native predict versus GPU max absolute difference={r['max_absolute_errors']['CPU_GPU_predict']:.9g}; fresh history encoder difference={r['max_absolute_errors']['CPU_GPU_history_encoder']:.9g}, both below fixed2e-4. Saved scalar/layout checks use at most1e-6 and are recorded individually. All32 saved arrays remain in two hashed G artifacts. No training was performed, no held-out forecasting benchmark was scored, and no causal truth was available in this fixture.

These facts justify correcting the implementation before capability comparisons. They do not establish a research gap, generalization failure, stochastic calibration failure or need for a new world-model architecture. The two sampled Bayesian branches and repaired branch share the same fitted context transitions; they are not three independent learned models.
''')
save('negative_evidence.md','''# Rejected interpretations

- A known-channel clamping bug is fixed by ordinary indexing; it is not a novel world-model mechanism.
- Mixing action and response coordinates is an implementation/layout defect. Correcting it is compulsory baseline maintenance, not evidence that a new representation objective is needed.
- Changing outputs after a repair does not prove improved forecasting or intervention validity. The arbitrary sine/cosine fixture has no physical ground truth.
- Whole-plan conditional completion can legitimately depend on later known covariates. Nonanticipation belongs to a different controlled-process contract; a conditional forecaster must not be refuted against an unstated guarantee.
- The hidden future+111 check passed exactly. A blanket future-response leakage claim is rejected for this exercised path.
- Native finite predictions and normalized transition rows establish execution, not calibration. The weighted fusion of sampled latent trajectories was exercised as implemented; no distributional quality was tested.
- Fixing centroid coordinate indices does not by itself align single-channel action embeddings with multichannel context embeddings. That remaining geometric question requires a valid task and appropriate existing controls; it is not yet a novel gap.
- Parameter bookkeeping and an incorrect wrapper-prefix assertion were errors in audit setup. The original scripts/receipts are retained with explicit corrections; they are not attributed to a model failure.
- The weight hash mismatch was a transport/storage incident. The final full hash matches public content and all loaded state values were checked. Its cause is unknown, and no hardware/driver diagnosis is asserted.
- C65's ordinary causal MDN remains a strong existing control. The generic assertion that conditional generators need a new causal world-model architecture is unsupported.
- Recent literature already discusses action identifiability, endogenous dynamics, delays, passive/forced separation and embedding-based control. Broadly restating those concerns is insufficient novelty.
''')
save('hypothesis_space.md','''# Competing explanations and next evidence requirements

| Explanation | Current test | Judgment |
|---|---|---|
| Hidden future response values leak through simulate | Fixed hidden+111 perturbation | Falsified for this exercised native path. |
| Known covariate is held fixed inside native rollout | Trace successive encoder inputs | Falsified in current code; ordinary clamp removes the discrepancy. |
| Action-centroid coordinates match action encoder coordinates | Sentinel and actual embedding layout | Falsified for the suffix slice; indexing-only repair executed. |
| These discrepancies require a new model | Minimal same-weight software repairs | Unsupported; exclude as novelty. |
| Corrected action embeddings provide useful transferable dynamics | No valid performance experiment yet | Open, not a positive finding or accepted direction. |
| A clinical intervention effect is identified | Arbitrary API fixture and qualitative published examples | Not tested; requires explicit identification assumptions and suitable truth. |

Before using a released framework as failure evidence, reproduce an appropriate supported task, preserve observed/known/hidden roles, and compare corrected native behavior with strong conventional dynamics. Distinguish approximation, training provenance, measurement normalization and unsupported tasks. Do not increase experiment count unless it can separate competing explanations that survive those controls.
''')
save('research_ideas.md','''# Research judgment after actual pretrained execution

Accepted final directions0. No new architecture is proposed. The operational capability now established is execution of a real released pretrained time-series framework, including its context-fitted transition branch, with explicit baseline repairs.

The worthwhile remaining audit target is whether reusable pretrained representations plus context-fitted transitions support a capability beyond ordinary forecasting on independently selected mechanisms. That broad target is not yet an idea: it lacks a specific unresolved failure beyond known inference, representation and causal-identification methods. A follow-up must first inspect existing original-task data/protocols and the already retained DoTime evidence, so it does not repeat old experiments or mistake unsupported synthetic transfer for original-model failure.

Reject software fixes, generic causalization, passive/forced decomposition, simple transition-density heads and ordinary Bayesian adaptation as contributions. The newly verified recent perspective and control papers further narrow these novelty boundaries. No final direction should be accepted until a task-valid failure survives strong applicable existing baselines and held-out validation.
''')
save('final_research_direction.md','''# No accepted direction yet

**Research Question:** Which reusable state/dynamics capability is still missing after baseline implementations and task semantics are correct? Current evidence does not identify a sufficiently novel answer.

**Evidence:** C65's causal MDN approaches the exact-law sampling reference on both tested systems. C66 loads and executes full pretrained NormWear-2, demonstrates two implementation discrepancies, executes ordinary corrections, and independently verifies numeric/layout artifacts. Neither finding proves a new framework is needed.

**Existing Methods:** Correct causal factorization, ordinary density estimation, known-input clamping, correct channel/latent indexing, passive/forced response structure, shared dynamical embeddings and context adaptation are established ingredients. Recent biosignal-world-model criticism also already covers delays, openness and endogenous evolution.

**Proposed Idea:** None accepted. A source defect or a task mismatch cannot be promoted into a publishable mechanism. The corrected real framework is an experimental platform, not the proposed method.

**Falsification:** Reject any candidate removed by ordinary software repair, consistent preprocessing, adequate training or an appropriate existing dynamics model. Do not claim failure on an unclaimed causal task. Reject positive results driven by test-set selection or unmatched information.

**Feasibility:** The complete186714897-parameter native model with187042577 state elements fits and runs on the RTX5090. Weights/data remain server G; authored code and compact evidence go to GitHub. Independent CPU/GPU checks and raw-array hashes are closed.

**Next Experiment:** Audit the original supported forecasting/controlled-channel task protocol and existing retained DoTime sources before freezing any additional evaluation. Select independent mechanisms and a supported original-task control, then assess corrected native pretrained inference against conventional train-only causal/dynamical baselines under explicit information constraints. A cross-benchmark transfer must be labelled as such. Do not launch another large suite until that comparison can falsify a specific surviving explanation. The earlier broad conditional-generation route is rejected as a novelty claim.

Overall research goal remains ACTIVE / UNACHIEVED; accepted final directions0. This cycle completes an implementation prerequisite, not the user's research-discovery objective.
''')
artifacts=['MODEL_FILES66_V1.json','WEIGHT_HEADER66_V1.json','INTERFACE66_V2.json','BAYES_INTERFACE66_V1.json','INDEPENDENT_REPLAY66_V1.json','TRANSPORT_REPAIR_FINAL66_V2.json','RAM_REPAIR_CHUNKS66_V2.json','interface_exit66_v1.json','interface_exit66_v2.json','bayes_exit66_v1.json','replay_exit66_v1.json']
dump(OUT/'CLOSURE66_V1.json',dict(utc=utc(),status='CLOSED_NATIVE_PRETRAINED_IMPLEMENTATION_AUDIT',reports=reports,artifacts={n:sha(OUT/n) for n in artifacts},accepted_final_directions=0,overall_goal='ACTIVE_UNACHIEVED',parameters=q['parameters'],state_elements=q['state_elements'],trained_models=0,performance_benchmarks=0,upstream_source_published=False,raw_artifacts='server_G_only'))
status='''# Current research status V66

Overall goal ACTIVE / UNACHIEVED. Accepted final directions0.

C65 is closed and published:18 native CSDI/ordinary MDN fits,240 generated laws,480 paired target views, independent replay PASS. Ordinary causal MDN do-distances0.078-to0.085 versus coverage-prefix CSDI0.223-to0.543 on the four fixed settings do not support a necessary new architecture. See cycle65 for seed-level results and runtime failures.

C66 executes a real pinned pretrained NormWear-2 framework: full277-tensor strict native load;186714897 parameters plus327680 buffers; native predict/simulate and conditional/unconditional latent transition paths. Full public weight SHA256 verified after a retained transport incident. Known-covariate recurrent overwrite and action-coordinate suffix mismatch are demonstrated and ordinary minimal repairs executed. Hidden future response perturbation has exactly0 effect. Independent array/layout replay and fresh native CPU/GPU checks PASS. See cycle66/experimental_results.md.

These are implementation prerequisites, not an original-paper dataset reproduction, held-out task result, causal-effect validation or new method. Software repairs are excluded as novelty. Recent primary work already discusses the generic action-validity and dynamics concerns. The next comparison must use corrected native baselines, supported task/data protocols, independently selected mechanisms and strong conventional dynamics. No final research direction is claimed.

All earlier experiments and unrelated projects are preserved. Raw arrays and checkpoints remain in authorized server G storage. GitHub receives compact code/reports/provenance; no result files are synchronized locally.
'''
with (PROJECT/'CURRENT_RESEARCH_STATUS_V66.md').open('x',encoding='utf-8') as f:f.write(status)
print(json.dumps(dict(status='CLOSED',reports=len(reports),closure_sha256=sha(OUT/'CLOSURE66_V1.json'))))
