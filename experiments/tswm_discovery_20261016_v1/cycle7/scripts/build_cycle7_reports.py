"""Private evidence ledgers: distinguish an objective witness from a new gap."""
import runtime_profile
import csv,datetime,json,pathlib,platform,sys
import numpy as np
import torch
ROOT=pathlib.Path(__file__).resolve().parents[1]

def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8'))
def write(p,text):
 path=ROOT/p;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text.strip()+'\n',encoding='utf-8')

PAPERS=[
 ('actswm','ActSWM','pure method',3,3,3,'Native source-default hinge and frozen inverse readout are the executed objects. The paper qualifies action separation by different consequences; the current task transfer does not refute its conditional theorem or reproduce Minecraft.'),
 ('slim','Keeping JEPA World Models Plannable','pure method',4,4,4,'Learned inverse dynamics and encoder gradients are strong existing controls. The paper includes action-null/shuffle, simulator ranking and loss-weight controls; do not claim these diagnostics are absent.'),
 ('mirabench','MiraBench','pure benchmark',3,4,None,'Action effectiveness, failure preservation and optimistic simulation already motivate a benchmark. A generic ineffective-action benchmark is insufficient differentiation.'),
 ('counterfactual_action_eval','Counterfactual Action Evaluation, Observation Bottlenecks, and Representation Geometry in Joint-Embedding Predictive World Models','method + benchmark',4,4,4,'Separates physical, observed, target-latent and predictor response. Public artifact permits checkpoint evaluation. Direct action-response auditing itself is covered.'),
 ('intervention_gap','The Intervention Gap in Latent World Models','method + benchmark',4,4,4,'Capture-gated matched interventions separate readable state from propagated effects, with action-null, wrong-action, probe and support controls. The current pilot is narrower and fully observed.'),
 ('dwm','DWM: Separating World Effects from Actions in Latent World Models','pure method',3,3,3,'Already separates autonomous world effects and action effects and tests persistent-drift variants plus zero-action data. Null motion versus autonomous motion is not an untouched distinction.'),
 ('caer','CAER','pure method',3,3,3,'Uses model-predicted action/null contrasts for training emphasis. Its predictive contrast is not claimed to identify causal graphs. Model self-contrast may merit calibration checks, but this pilot did not execute CAER.'),
 ('flex_wam','FLEX-WAM','pure method',3,3,3,'Action/state sensitivity balance uses a model-relative elasticity target. It differs from a universal positive separation margin and was not executed here.'),
 ('sensorimotor','Sensorimotor','pure method',3,3,3,'Inverse conditional information, state/action structure and homomorphism foundations are relevant existing mechanisms. Abstract and method were screened; no implementation or numerical claim is verified by this cycle.'),
 ('action_homomorphism_ijcai2003','State/action homomorphisms, IJCAI 2003','theory/proof',4,3,None,'State-dependent equivalent actions and action abstraction are old concepts. A useful residual must concern estimation or generalization under a precise modern training protocol.'),
 ('dotime','DoTime','method + benchmark',4,4,3,'Temporal intervention generator, versioned suites and native protocol inspected. Its public HF card reports the observational-target erratum: the original direction advantage does not survive correction. No PFN weight execution in this cycle.'),
 ('causal_longpfn','CausalLongPFN','pure method',3,3,3,'Existing planned longitudinal-treatment PFN baseline. Paper screening only; no checkpoint inference or reproduction claimed.'),
 ('csa_pfn','Amortizing Causal Sensitivity Analysis with Prior-Fitted Networks','pure method',4,3,3,'Amortized sensitivity bounds already exist for static treatment tasks. A generic partial-identification PFN is not new; temporal distinctions need comparison with longitudinal sensitivity literature.'),
 ('counterfactual_debugging','Counterfactual Debugging of World Model Transfer','method + benchmark',4,4,4,'Mechanism replacement and Shapley attribution use queryable world/environment mechanisms with explicit sparsity and interaction assumptions. No source execution here; do not equate its independent world noise with shared-noise counterfactual coupling.')]

def main():
 rows=read('outputs/action_equivalence_pilot/results.json');assert len(rows)==27
 status=read('outputs/action_equivalence_pilot/status.json');assert status['state']=='complete'
 cert=read('outputs/objective_certificate/result.json');assert len(cert['rows'])==6
 lit=read('outputs/literature/registry.json');assert len(lit['papers'])==14 and not lit['errors']
 pin=read('sources/actswm/SOURCE_PIN.json')
 for row in rows:
  log=ROOT/'outputs/action_equivalence_pilot'/f"{row['kind']}_{row['arm']}_s{row['seed']}"/'train.jsonl'
  entries=[json.loads(s) for s in log.read_text().splitlines()]
  assert len(entries)==6 and entries[-1]['step']==1500
  assert row['trainable_parameters']==15312 and row['frozen_IDM_parameters']==267777
  assert np.isfinite([v for e in entries for v in e.values()]).all()
 summaries=[]
 keys=['factual_MSE','paired_response_MSE','predicted_cosine_gap','true_cosine_gap','null_response_RMS','offline_candidate_regret_mean']
 for kind in ['null','linear','deadzone']:
  for arm in ['prediction','hinge','full']:
   subset=[r for r in rows if r['kind']==kind and r['arm']==arm];assert len(subset)==3
   r={'kind':kind,'arm':arm,'initialization_seeds':[31,32,33]}
   for k in keys:
    vals=[s['test_final'][k] for s in subset if k in s['test_final']]
    if vals:r[k]={'mean':float(np.mean(vals)),'sample_sd':float(np.std(vals,ddof=1)),'values':vals}
   summaries.append(r)
 write('outputs/action_equivalence_pilot/summary.json',json.dumps(summaries,indent=2))
 table='| Dynamics | Loss arm | Factual MSE mean ± SD | Response MSE | Predicted / true cosine gap | Null response RMS | Candidate regret |\n|---|---|---:|---:|---:|---:|---:|\n'
 for r in summaries:
  val=lambda k:r[k]['mean']
  null=f"{val('null_response_RMS'):.6g}" if 'null_response_RMS' in r else 'N/A'
  table+=f"| {r['kind']} | {r['arm']} | {val('factual_MSE'):.6g} ± {r['factual_MSE']['sample_sd']:.3g} | {val('paired_response_MSE'):.6g} | {val('predicted_cosine_gap'):.6g} / {val('true_cosine_gap'):.6g} | {null} | {val('offline_candidate_regret_mean'):.6g} |\n"
 certtable='| Frozen readout seed | Arm | Train-selected spurious force | Exact predictor test objective | Wrong predictor test objective | Wrong predictor test factual MSE |\n|---|---|---:|---:|---:|---:|\n'
 for r in cert['rows']:
  a,b=r['true_vs_perturbed'];assert a['delta']==0 and b['test_loss']<a['test_loss'] and b['test_factual_MSE']>0
  certtable+=f"| {r['seed']} | {r['arm']} | {r['train_selected_delta']:.3g} | {a['test_loss']:.7g} | {b['test_loss']:.7g} | {b['test_factual_MSE']:.7g} |\n"
 scope='''This is a **new controlled mechanism experiment**, not an original-paper Minecraft reproduction, not a real-data forecasting result and not a demonstrated new method. Unchanged native class/function bodies are loaded by AST; the model is deliberately reduced, and the physical-state encoder is fixed to identity. The learned encoder, pixel information bottleneck and native SIGReg gradients are outside scope. No existing source or earlier sealed cycle was modified.

Three 4D deterministic plants use x_next=.8*x+[0,.04,-.03,.02]+force(a)*e0. Force is zero, .35*a, or .35*sign(a)*relu(abs(a)-.5). Actions are randomized independently of state, with 30% exact zero and 70% uniform[-1,1], including during training. Train/validation/test counts are 8192/1024/2048. Cohort seeds are 15001/15002/15003 plus 100*plant_index. Three initialization seeds share these cohorts: they measure initialization variability, not independent dataset replication.

The native ARPredictor is reduced to width32, depth1, heads2, context1 and horizon3; 15,312 trainable parameters. All arms instantiate the same 267,777-parameter frozen random native inverse readout; only full uses its loss. Native-default AdamW1e-4, wd1e-3, clipping1 and 1,500 updates of batch128 are held fixed. Arms are prediction only; prediction+hinge(weight.5,margin.3); and prediction+hinge+frozen-IDM(weight1). Fixed-step final is primary; minimum factual-validation MSE checkpoints are secondary, with no test selection. Native prediction loss is independently matched to the evaluator before every fit. Native alternate-action rollout retains the first context action and zeroes later actions; its first-step hinge is therefore a constant, disclosed rather than silently corrected.

Cosine separation is evaluated in the fixed physical coordinates; no claim of representation-coordinate invariance is made. Null-plan sanity checks assert exact equality of physical trajectories for action*.4 in null/deadzone plants. Candidate regret compares the chosen plan with the true best among a fixed bank of 64 plans at 256 held-out initial states. This is offline candidate ranking, not closed-loop MPC; the physical oracle is never used for training or checkpoint selection.'''
 write('README.md',f'''# Cycle 7: action-equivalence objective audit

Scientific jobs complete; overarching discovery goal remains **ACTIVE and unachieved**. No direction is accepted for publication development.

27 matched fits and 6 finite objective witnesses executed on the remote RTX5090. Source [ActSWM](https://github.com/KeithBlackmore/ActSWM-) pinned to `{pin['revision']}`. Fourteen primary PDFs archived with hashes.

{scope}

The default auxiliary objectives amplify false effects in the tested plants. A perfect physical predictor can have worse native total objective than a deliberately wrong predictor. However, prediction-only already solves these fully observed tasks, and existing learned inverse models/action abstractions are relevant controls. This result supports a scoped objective mismatch, not a novel general research problem.

See experimental_results.md, hypothesis_space.md and final_research_direction.md. Runtime/data/source/checkpoint hashes are included in the artifact manifest after sealing. All content remains private.
''')
 write('experimental_results.md',f'''# Completed remote experiments

## Protocol and scope

{scope}

## Fixed-step held-out results

{table}

Mean ± sample SD across three initialization seeds; no independent-cohort confidence interval or population significance claim. Raw per-fit metrics include secondary validation-selected checkpoints in outputs/action_equivalence_pilot/results.json. Each fit retains best.pt, final.pt and six raw training-log records. Total training time {sum(r['seconds'] for r in rows):.2f} seconds; process orchestration and evaluation are additional.

## Finite objective witness

The exact known all-null plant is augmented by delta*a*e1. A fixed121-point grid [-.6,.6] selects delta by the native TRAIN objective on4096 fresh samples; validation2048/test4096 use fresh seeds61002/61003. Seed61001 generates training. This is a privileged diagnostic, not a learned baseline, theorem or population certificate. The same cohorts are reused across frozen readout seeds.

{certtable}

All six held-out witnesses lower the unchanged source-default objective by introducing physically wrong force. This rules out optimization failure and insufficient model capacity as the sole explanation for this objective-level mismatch. It does not show that all validation-tuned weights fail, or that a learned/probabilistic inverse model must fail.

## Non-executed objects

DoTime PFN repository metadata only: selected checkpoint295,027,515 bytes, revision bec6618c8b25f3a5d8cf608d4c597978d5a694ac, SHA256214db2fab16fb5fa17e68244abc0df3b7b0154b9fb2d2ce16b2be3f4d5d93618. Weights were not downloaded and no PFN accuracy result is claimed. Exact metadata is in outputs/dotime_pfn_metadata.json. SLIM, DWM, MiraBench, CAER and FLEX-WAM were read, not run.
''')
 write('baseline_registry.md',f'''# Baseline and provenance registry

| Item | Status | Exact scope |
|---|---|---|
| ActSWM source | Pinned/read/executed unchanged selected bodies | {pin['revision']};17 small source/config files, no native data/weights |
| Native ARPredictor + loss + frozen IDM | 27 completed fits | New 4D fully observed plants, smaller native architecture; not Minecraft reproduction |
| Prediction-only | Executed stop/attribution control | Same architecture/init/train data/updates; fewer auxiliary computations; already solves the pilot |
| Perfect plant + spurious force | 6 finite held-out objective witnesses | Privileged oracle diagnosis; not a trainable or fair forecasting baseline |
| Learned/probabilistic IDM, SLIM | Not executed | Required before a general inverse-dynamics failure claim |
| DoTime PFN | Metadata only | HF mirror exact revision and single checkpoint size/hash;55 checkpoints in repository, no weights downloaded |
| Earlier models/cycles | Read-only existing evidence | Preserve sealed main/cycles2–6; do not count their results as fresh execution |

Data are generated from saved explicit DGPs; compressed cohorts and SHA256 descriptors are retained. Train/validation/test use independent cohort seeds. Source pin contains per-file hashes; native source bodies are not patched. Adapter code, plans, runtime profile, raw metrics and checkpoints are retained. Python{platform.python_version()}; full runtime versions and process exits are recorded separately. No public upload is authorized for this research cycle.
''')
 write('negative_evidence.md','''# Negative evidence and rejected claims

1. **Larger predicted action separation means better dynamics: rejected within this pilot.** Hinge/full raise separation but degrade physical response error and candidate ranking on all three plants, including a positive linear-effect control.
2. **Only missing zero-action support explains the null failure: rejected.** Training explicitly includes30% exact-zero actions. Actions are randomized independently of state.
3. **Only optimization or capacity failure explains it: rejected as a sole explanation.** A privileged exact-plant witness has a lower held-out default objective after adding false force.
4. **This establishes a new method need: not supported.** Ordinary prediction-only is already accurate on every fully observed plant. Removing a damaging auxiliary term is a necessary existing baseline, not an innovation.
5. **Every inverse-dynamics objective must fail: not supported.** Frozen random inverse readout differs from learned/probabilistic inverse models; state-dependent action equivalence and homomorphisms are old. A Bayes-optimal inverse conditional can represent action ambiguity rather than invent state differences.
6. **Native Minecraft result/theorem refuted: not supported.** The paper qualifies separation for distinct consequences. Architecture/input/domain are changed and no native checkpoint or Minecraft data are evaluated.
7. **Default-weight transfer failure proves all tuned variants fail: not supported.** No loss-weight validation sweep was run. Existing source-default values are only a defined diagnostic condition.
8. **General action-null/capture audit is novel: rejected as a broad novelty claim.** MiraBench, Counterfactual Action Evaluation and The Intervention Gap already contain direct action-effect/failure/capture diagnostics. DWM addresses autonomous versus action-driven change.
9. **DoTime original interventional direction advantage is a valid baseline target: rejected by the authors' own erratum.** HF model card says corrected observational targets remove that gap under the paper protocol; no new rediscovery claim is made.
10. **New amortized causal-bound PFN is novel by itself: rejected.** CSA-PFN already amortizes sensitivity bounds, and longitudinal sharp-bound methods predate this cycle. Specific temporal residuals remain open questions, not accepted ideas.

Stop decision: do not launch a new action-gating architecture or a default-weight grid solely to turn this controlled negative result into a positive paper. Further work needs a real temporal task and failure surviving appropriate learned/probabilistic inverse models, accurate ordinary predictors, correct observation/history interfaces and validation-selected hyperparameters.
''')
 write('hypothesis_space.md','''# Competing hypotheses and falsification

| Hypothesis | Discriminating evidence | Current judgment |
|---|---|---|
| H1: source-default positive-separation objective rewards false effects | Randomized null/deadzone task; unchanged native loss; exact-plant perturbation witness | Supported locally,3 initialization seeds and6 held-out witnesses |
| H2: absence of no-op training causes all failure | Put exact-zero actions in training support | Rejected as sole explanation |
| H3: optimizer/capacity is the sole cause | Perfect physical model vs wrong perturbation under same total loss | Rejected as sole explanation |
| H4: improving action sensitivity improves decisions | Independent paired physical effects and fixed candidate-bank regret | Rejected locally; not a closed-loop or native benchmark claim |
| H5: action equivalence needs a new mechanism | Existing prediction-only, learned/probabilistic IDM and action abstraction controls | Not supported; prediction-only is sufficient in current task |
| H6: the residual matters under learned state compression and native noise | Learned encoder/native benchmark with real temporal interface and validated baselines | Untested; must establish before method design |

An eligible next experiment must freeze new cohort identities before evaluating any candidate. Compare existing learned inverse likelihood and prediction-only under equal information and matched capacity/budget. Include positive-effect, null/deadzone and observational-alias controls. If an existing properly specified inverse model or standard predictor removes the failure without losing the positive-effect capability, reject new-method necessity. If the effect exists only at a poor default weight, treat it as a tuning boundary. A native learned-encoder failure alone does not establish novelty; compare the closest public diagnostics and abstraction objectives.

No test data choose thresholds, representation dimension, method family or loss weights in this cycle. Reported seed SD is initialization-only. No claim that absence of a quick literature match proves novelty.
''')
 write('research_ideas.md','''# Candidate questions, not accepted directions

Only one provisional question is retained; zero methods are recommended for submission development.

**Conditional action-effect calibration under state-dependent action equivalence.** Can an existing compact temporal world model preserve useful small effects while assigning negligible response to physically equivalent action histories, with independent prospective validation rather than maximizing its own action contrast?

Direct evidence: current native-objective probe demonstrates false sensitivity and ranking harm under default auxiliary weights. Missing evidence: learned-encoder/native-task failure that survives ordinary accurate predictors and learned/probabilistic inverse models. Existing literature already covers action abstraction, inverse objectives, physical-response audits and failure preservation. A potential residual would have to concern statistically reliable conditional effect estimation under partial observability; that residual is inferred and low confidence, not established.

No proposed new architecture is justified yet. The economical continuation is a native temporal benchmark/control experiment only after checking its source/interface and the recent closest papers. If standard controls solve it, retire this candidate rather than rename their mechanism.

Do not restart generic long-loop tuning, scalar uncertainty fixes, linear covariance-state toys, simple Gaussian coupling solvers or CHiLD density-head tuning from earlier cycles.
''')
 write('final_research_direction.md','''# Current research decision

**No final direction has passed the user's innovation and evidence requirements. Goal remains ACTIVE.**

**Research Question:** provisional only—when do action-discriminative training objectives confuse command distinguishability with physically distinct future effects in compact temporal world models?

**Evidence:**27 remote matched fits across zero, linear and deadzone dynamics, plus6 exact-plant objective witnesses. Default native losses induce false response and larger offline candidate regret. All facts are scoped to the fixed full-state representation and reduced model.

**Existing Methods:** action homomorphisms, learned inverse dynamics, SLIM, DWM and recent intervention/capture benchmarks already address substantial parts. Prediction-only already solves the current fully observed task. Existing methods have therefore NOT yet been shown insufficient.

**Proposed Idea:** none accepted. Do not invent a new gate by renaming inverse uncertainty or action equivalence. A possible future mechanism would require evidence beyond standard correctly specified inverse likelihood and ordinary state predictors.

**Falsification:** abandon this as a new-method direction if validation-tuned standard predictors or learned/probabilistic inverse models preserve the positive-effect capability and remove spurious responses; also abandon if failure is only a default-weight transfer artifact or the nearest literature already covers the precise protocol.

**Feasibility:** native ActSWM source is pinned and small controlled runs complete on one RTX5090. Learned-encoder/native temporal-task controls are feasible but have not been executed. Fourteen primary documents and source/data/config/checkpoint provenance are retained. DoTime PFN weights remain undownloaded; the original observational comparison has a public erratum.

**Next Experiment:** first establish an eligible native temporal task with independently measurable action equivalence and positive effects, then freeze development/test families and compare existing learned inverse likelihood, prediction-only and the released objective. Require a failure surviving these controls before proposing any method. The current toy alone is insufficient to justify a top-conference research direction.
''')
 registry={r['name']:r for r in lit['papers']}
 search='literature-search-20261016-action-equivalence'
 paperrows=[]
 for key,title,typ,ins,complete,num,claim in PAPERS:
  p=registry[key];url=p['url'];venue='IJCAI2003 proceedings' if key=='action_homomorphism_ijcai2003' else ('author-hosted primary PDF; acceptance not independently used' if key=='counterfactual_debugging' else 'arXiv2026 pinned preprint; venue acceptance not asserted')
  paperrows.append({'key':key,'title':title,'type':typ,'venue_status':venue,'url':url,'insight_1to5':ins,'completeness_1to5':complete,'numeric_evidence_1to5':num if num is not None else 'N/A benchmark/theory','reading_scope':'Selected method/protocol sections and relevant appendices; not exhaustive formal proof verification','code_executed':key=='actswm','evidence':claim})
 (ROOT/search).mkdir(parents=True,exist_ok=True)
 with (ROOT/search/'papers.csv').open('w',encoding='utf-8',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(paperrows[0]));writer.writeheader();writer.writerows(paperrows)
 papertext='# Exploratory primary-source screening\n\nScores are provisional judgments of inspected evidence, not independent reproduction scores. Benchmark/theory papers receive N/A for experimental numeric scoring. Code execution is claimed only for selected ActSWM bodies.\n\n'
 for p in paperrows:papertext+=f"- [{p['title']}]({p['url']}) — {p['type']}; {p['venue_status']}. Insight/completeness/numeric: {p['insight_1to5']}/{p['completeness_1to5']}/{p['numeric_evidence_1to5']}. {p['evidence']}\n"
 write(search+'/papers.md',papertext)
 write('literature_review.md',papertext+'''

## Closest-work clusters and residuals

Action-discriminative training: ActSWM, SLIM, Sensorimotor, FLEX-WAM. Covered: inverse/action information and sensitivity repair. Under-tested in this cycle: a validated learned encoder under conditionally equivalent actions. Possible differentiation: finite-data effect calibration under state aliasing, only if existing probabilistic inverse controls fail.

Physical fidelity and autonomous effects: MiraBench, Counterfactual Action Evaluation, Intervention Gap, DWM, CAER. Covered: failure preservation, physical-vs-latent effects, capture gates, action/null contrasts and autonomous dynamics. Residual candidate: objective-level attribution under an exactly specified task; not itself a new benchmark or new method.

Causal amortization: DoTime, CausalLongPFN, CSA-PFN. Covered: interventional priors, longitudinal planned treatments and amortized sensitivity bounds. Public erratum is an essential protocol anchor. A temporal identified-set problem must be distinguished from forecast quantiles and existing longitudinal sensitivity theory before implementation.

No closest-paper title is a kill gate. The rejection here concerns overly broad novelty claims and the already-successful ordinary control; precise unexplored residuals remain hypotheses.
''')
 write(search+'/search-notes.md','''# Search log and policy

Mode: exploratory;14 final screened primary papers in5–10 thematic clusters/subclusters, plus near-miss PFN/causal-bound routes. Search date2026-10-16; recent-paper focus2026April–October, foundations back to2003. All archived PDFs version-pinned except author-hosted PDF, which is SHA256-pinned.

Public queries included time series world model intervention nonidentifiable partial identification2026; world model action null inverse dynamics2026; exact public titles DWM and Intervention Gap; WorldKernel counterfactual honesty; temporal world model hidden confounding sensitivity. No private result text or unpublished hypothesis was sent as a search query.

Discovery indexes led to arXiv, official proceedings, author pages and public source. Secondary summaries were discovery only. Policy-excluded MDPI/preprints.org material was not selected. Acceptance labels from secondary pages were not promoted to verified venue status. Papers are deduplicated by archive/title. Source hashes and retrieval timestamps are in outputs/literature/registry.json.

Native repository inspection: ActSWM exact commit,17 small files; no AGENTS instructions found in tree. Native rollout/predictor/IDM bodies executed; launcher bypassed. DoTime current public model card and pinned local native protocol inspected; mirror metadata uses an exact revision and is not assumed to be official latest. No full HF weight repository download.

Additional verified primary protocol source: https://huggingface.co/thummd/do-over-time-pfn reports s9 observational zero-target erratum and corrected s10 arms; original matched direction-accuracy gap does not survive correction. Related WorldKernel position paper already advocates counterfactual intervals; generic honesty/partial-identification is not an untouched question.

Limit: selected sections were read, not every proof/reference independently audited. Quality scores are provisional fit/evidence judgments, not universal paper rankings. No novelty or top-conference readiness claim is made.
''')
 write(search+'/idea-grounding.md','''# Evidence-to-idea boundary

Source-supported primitives: positive action-separation margins; fixed versus learned inverse maps; state-dependent action equivalence; capture-first physical-response evaluation; autonomous/action decomposition; temporal causal priors; sensitivity bounds.

Experimental observation: default native objective prefers some false physical effects and harms candidate ranking in a controlled fully observed setting. Ordinary prediction-only succeeds. The objective witness rules out sole optimization/capacity explanations within this task.

Inferred residual: reliable conditional action-effect calibration under state aliasing may not follow from global action sensitivity. Confidence LOW: neither insufficiency of existing learned inverse models nor novelty beyond recent capture/response audits is established.

Protocol anchors: exact-zero support; independent randomized actions; distinct train/validation/test cohort seeds; native action timing; fixed-step primary; explicit native-vs-transfer scope; exact oracle not counted as learned baseline; initialization SD not dataset replication; corrected DoTime observational arm.

Conflict: forcing command separation can disagree with physical equivalence, while existing work seeks to prevent action ignoring. The meaningful question concerns conditional evidence for distinct consequences, rather than universally increasing or decreasing sensitivity. Minimum experiment must include positive and null effects with an existing ambiguity-aware inverse likelihood. No candidate method is endorsed before that test.

Search handoff remains exploratory. Do not mark the entire area solved because related titles exist; equally, do not call an ordinary removed regularizer an innovation.
''')
 task={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cycle7':{'scientific_jobs_complete':True,'fits':27,'objective_witnesses':6,'primary_pdfs':14,'accepted_research_gap':False},'objective_achieved':False,'goal_status':'active','decision':'Controlled objective mismatch supported; ordinary baseline sufficient; no final direction accepted'}
 write('outputs/task_status.json',json.dumps(task,indent=2))
 write('outputs/runtime.json',json.dumps({'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'torch':torch.__version__,'cuda':torch.version.cuda,'GPU':torch.cuda.get_device_name(0),'profile':'scripts/runtime_profile.py','preserve_earlier_cycles':True},indent=2))
 print(json.dumps(task))

if __name__=='__main__':main()
