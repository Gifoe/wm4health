"""Evidence checkpoint, not a declaration that autonomous research is complete."""
import csv,datetime,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
def write(name,value):
 path=ROOT/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(value.strip()+'\n',encoding='utf-8')
def main():
 inverse=json.loads((ROOT/'outputs/inverse_score/results.json').read_text(encoding='utf-8'))
 metrics=json.loads((ROOT/'outputs/dotime_full/metrics.json').read_text(encoding='utf-8'));diag=json.loads((ROOT/'outputs/dotime_full/diagnostics.json').read_text(encoding='utf-8'))
 pooled=[x for x in metrics if x['structure']=='ALL']
 table='| Model | Full-suite RMSE | Effect-sign accuracy | Scored effects |\n|---|---:|---:|---:|\n'+'\n'.join(f"| {r['model']} | {r['rmse']:.6f} | {r['dir_acc_effect']:.6f} | {r['dir_n_valid_effect']} |" for r in pooled)
 scoretable='| Seed | Native orientation mass | Generic quadratic orientation mass | Native excess nats | Quadratic excess nats |\n|---|---:|---:|---:|---:|\n'
 for seed in [2021,2022,2023]:
  native=next(r for r in inverse['rows'] if r['seed']==seed and r['arm']=='native_distance');quad=next(r for r in inverse['rows'] if r['seed']==seed and r['arm']=='generic_quadratic')
  scoretable+=f"| {seed} | {native['test_correct_orientation_mass']:.6f} | {quad['test_correct_orientation_mass']:.6f} | {native['test_excess_over_class_optimum']:.6f} | {quad['test_excess_over_class_optimum']:.6f} |\n"
 write('experimental_results.md',f'''# Cycle9 experimental evidence

All numeric experiments ran on the private RTX5090 server. Original cycles1–8 were preserved.

## Native inverse-score controlled transfer

{scoretable}

Six fits,600updates each; seeds2021/2022/2023, independently rotated train/validation/test angle grids. Native function `action_discrimination_loss` was extracted unchanged by AST from pinned AC-MTM `train.py`. Endpoint inputs contain only loop orientation. Each endpoint class contains20 indistinguishable held-out rotations; the correct index loss floor is log20, not zero. A correct orientation score can therefore have only5% index accuracy. Native score has16 free query parameters; generic degree-two feature score has72. This is an expressivity/known-repair diagnostic, not a capacity-matched method superiority claim or a native visual-control replication.

At zero query, native gradients are at most6.62e-8; tested random query perturbations increase loss. More importantly, row-constant cancellation gives a convex logsumexp objective; with balanced opposite rotations the class-conditional action mean is zero. The zero query is therefore a population/finite-balanced-bank minimizer regardless of an endpoint MLP's capacity. The test establishes a restricted-score boundary, not failure of all contrastive inverse learning. A numerical Hessian was not needed or computed; convexity is an analytic argument.

ACRO's released continuous-action pretraining normalizes action labels and predicted actions. Collinear positive actions1,2,0.25 normalize identically. This is an algebra/code contract, not a full ACRO training result. EP-IDM was source-audited only; its linear theorem and explicitly acknowledged endpoint-rank limitation are not contradicted by a nonlinear square-loop construction.

## Published DoTime checkpoint, official corrected suite

{table}

Official `dot-Identifiability-v1`1.2.0:12,150episodes,9structures,1350perstructure. Native loader verified released manifest MD5s; shared-noise and target QA passed all12,150episodes. Level-sign denominator10,235 differs from effect-sign denominator4356. Near-zero effects are excluded by the native threshold0.1; raw level errors use every episode. `bow_graph` is included in the pooled descriptive result but is explicitly not nonparametrically identified. These accuracies are not guarantees of causal identification.

Checkpoint:`s9ho_all_causal/do_over_time_pfn_best.pt`, HF revision`bec6618c8b25f3a5d8cf608d4c597978d5a694ac`, SHA256`214db2fab16fb5fa17e68244abc0df3b7b0154b9fb2d2ce16b2be3f4d5d93618`,295,027,515bytes. It loaded24,922,633parameters strictly with native `load_dotpfn`; static checkpoint inspection found no unsafe pickle globals. It is the original published s9 causal model, **not the later s13 or redesigned recipe**. Later s10/s13 artifacts are visible in the retained metadata but were not executed here. No interventional-vs-observational training superiority is inferred.

Native `PFNRef` preprocessing/prediction is preserved. Batched inference agrees with8native scalar calls within{diag['native_scalar_vs_batch_max_difference']:.3g}; native final scorer is also retained. One checkpoint, no training, no hyperparameter selection. The fixed540episode subset was followed by one full-suite pass to remove sampling uncertainty. Full prediction/scoring took{diag['elapsed_seconds']:.2f}s after preparing/loading the suite. Peak allocated CUDA memory{diag['peak_cuda_allocated_bytes']/1024**2:.1f}MiB. This excludes source/data downloads and parsing.

The nominal90%interventional outcome interval covers{diag['coverage']:.6f}; mean width{diag['width_mean']:.6f}; no adjacent quantile crossings. These are outcome-level quantiles, not individual counterfactual-effect intervals. NaiveOLS is an unadjusted associative control with omitted-variable bias, not a causal oracle. Mean and VAR-OLS are action-blind controls; grouping is explicit.

This poor s9 result has sufficient existing explanations. The author's pinned erratum/re-test ledger already withdraws the interventional advantage and documents normalization, role/addressing and temporal aggregation repairs. Our execution confirms an available artifact; it does not discover those defects or establish a new gap.

Raw logs:`outputs/inverse_score/*log`,`outputs/dotime/*log`,`outputs/dotime_full/*log`. Predictions/configs/metrics are retained separately. A metadata-print wrapper initially used `suite.metadata` instead of native `suite.meta`; it failed after the download completed. The corrected wrapper reused the cached verified data. No scientific training job was restarted to address a timeout.
''')
 write('baseline_registry.md','''# Cycle9 baseline registry

| Artifact | Revision / version | Execution status |
|---|---|---|
| LeoToso/JEPA_control |6c3163e2f597a1be75f5dfc0d2836793b821968d|Selected source/config audit only; no native training/control result|
| manantomar/agent-centric-representations |f4622fadcb1fa2b6724dbab21ee62d1ac250bfe4|Selected source audit plus normalization algebra; no RL reproduction|
| jackboyla/action-contrastive-jepa |43c96f66bb12c95208be68b0858ee658c7e1d363|Unchanged native loss function executed on a new controlled diagnostic; not a native task reproduction|
| Generic quadratic action score |cycle9 script|Existing degree-two feature/InfoNCE control;3seeds,600updates; not presented as new method|
| thummd/dotime |8f227406a5b17c2040cfdcc680b493729ef47d00|Native generator-release loader, preprocessing, PFN and CPU baseline code executed;73native dependency/protocol files retained byte-identically|
| HF thummd/do-over-time-pfn |bec6618c8b25f3a5d8cf608d4c597978d5a694ac|One original s9ho_all_causal checkpoint downloaded/verified/executed; later corrected s13 checkpoints inventoried, not executed|
| dot-Identifiability-v1 |1.2.0|Official12,150episode release; manifestMD5 and additionalSHA256 retained|
| Mean / VAR-OLS |same native source|Action-blind controls, complete suite|
| NaiveOLS |same native source|Same observed-history/do-value information as PFN; unadjusted regression, complete suite; not causal oracle|
| pfns |0.4.2|Published wheelSHA verified; no-deps installation into cycle-local vendor directory; earlier environments untouched|

Source snapshots of the3inverse-learning repositories are explicitly selected-file snapshots with complete tree inventories. No claim of complete native environment replication. No original launcher executed. Runtime profile uses process-local single-thread numeric libraries and the previously verified affinity. Source/protocol retention and package licenses are recorded. All experiment selection and metric denominators are in configs/results.
''')
 write('negative_evidence.md','''# Cycle9 negative evidence

1. **New generic multistep inverse objective:** not supported. ACRO/AC-State precede EP-IDM; the new native repository also contains multistep variants. A theorem's stated linearity/rank limit is not an undisclosed flaw.
2. **All contrastive inverse objectives preserve the relevant nonlinear information:** not established. The tested raw-action quadratic-distance score stays at chance on balanced loops. Ordinary quadratic-feature contrast succeeds, so the broad new-method need is rejected on this diagnostic.
3. **Our quadratic score is an innovation:** rejected. Polynomial features, generic density-ratio contrast and conditional kernel embeddings are existing tools. The6fits do not justify a new architecture or paper direction.
4. **Original DoTime PFN low effect accuracy is a newly discovered world-model gap:** rejected. The author's current source already records the failure and recovery programme. One original frozen checkpoint also cannot determine the performance of later corrected training recipes.
5. **NaiveOLS accuracy proves causal identification:** rejected. Unadjusted regression is biased under hidden confounding; finite-prior predictive accuracy differs from nonparametric identification.
6. **Forecast/invariant-statistics fidelity versus forced-response fidelity is a new broad claim:** already directly studied by the October5response-theory probe and older response-theory work. A specific residual must be differentiated before execution.
7. **This cycle completes autonomous discovery:** false. It provides reproducible elimination evidence and a full additional native baseline. No final research direction meets the user's standard yet.
''')
 write('hypothesis_space.md','''# Cycle9 competing hypotheses

| Hypothesis | Discriminator | Outcome | Decision |
|---|---|---|---|
| Loop endpoint contains no useful action information |Known nonlinear plant + generic degree-two score, held-out rotations|Orientation is present; generic score mass>.9945|Information absence rejected for this constructed diagnostic|
| Native score fails only from insufficient head optimization/capacity |Free class queries, zero gradient, convex logsumexp derivation|Global balanced-bank optimum is chance|Restricted score family explains it; not a training-budget failure|
| Inverse contrast inherently cannot use the information |Generic quadratic contrast control|Excess loss~.0054nats vs native log2|Broad inherent-failure claim rejected|
| Long-horizon inverse preservation is unstudied |ACRO/AC-State/action-bisimulation and native configs|Closest work already studies it|No generic novelty claim|
| DoTime performance is a wrapper/loader failure |Hash, strict native model load, native target QA, scalar/batch equivalence, full native scoring|All pass|Available s9 artifact is genuinely weak under this corrected release protocol|
| That s9 weakness is an unsolved innovation opportunity |Author erratum, s13/recovery source, strong NaiveOLS control|Known defects and an existing simple control suffice|Do not promote it to new research gap|
| All meaningful time-series world-model questions are solved |None of the above tests can establish this|Unresolved|Keep the broad objective active; these are local eliminations|

Next discriminator must target a specific residual beyond a known defect. A provisional audit question is whether temporal history provides informative constraints on causal effects under the exact released hidden-confounder model class, and which uncertainty remains prior-driven. This is **not** a claim that a bow graph becomes nonparametrically identified from history: fresh hidden innovations and model-class assumptions must be handled explicitly. Causal time-series ID algorithms, proxy/negative-control methods and ordinary latent state-space likelihoods are mandatory existing controls. No new method or new simulation cohort is frozen yet.
''')
 write('research_ideas.md','''# Cycle9 candidate status

**No research direction is endorsed.** The strongest completed result is failure attribution, not novelty.

One provisional next question, low confidence: can a time-series world model separate effect information constrained by temporal observations from effect information supplied by its training prior, under latent confounding and finite histories?

The concrete anchor is the now-executed corrected DoTime1.2.0release, with identified structures and an explicit nonidentified bow control. The source-level question is what the exact temporal graph and parametric generator restrict, rather than treating every hidden-confounded episode as impossible or treating finite-prior accuracy as causal identification. This requires a formal identification audit before any method investment. Existing causal-ID-on-time-series, latent state-space inference, proxy methods, sensitivity analysis and causal PFNs may already cover the useful cases. It is not currently a publication claim.

Minimal next test: inspect the exact native temporal mechanism/noise constraints; construct or rule out observationally equivalent mechanisms with different intervention effects within that declared class. If ordinary likelihood/proxy inference or existing identification theorems settle the issue, record that and change scope. Do not train a new PFN merely to improve on the already-diagnosed s9 model, and do not relabel a generic posterior/sensitivity method as new.

Stopped local claims: new generic inverse contrast, new multistep inverse preservation, new forecast-versus-response distinction, and newly discovered s9 training defects. Their broader fields remain open; only these unsupported novelty framings are eliminated.
''')
 write('final_research_direction.md','''# Current research decision

**The full autonomous research goal remains ACTIVE and UNACHIEVED. No final direction is recommended.**

**Research Question:** not finalized. A provisional next audit concerns the boundary between temporal-data evidence and prior-dependent causal-effect predictions under hidden confounding.

**Evidence:** native Action-NCE function executed in6controlled fits; ordinary quadratic contrast resolves its information limitation. One published24.9Mparameter DoTime model was strictly loaded and evaluated on all12,150official1.2.0episodes with native controls; PFN effect-sign53.81%, NaiveOLS77.32%. Both outcomes have known-method/known-defect explanations and do not independently establish novelty.

**Existing Methods:** ACRO/AC-State/action-bisimulation, generic conditional density-ratio contrast, causal time-series identification, ordinary latent state-space likelihoods, proxy/sensitivity methods and the DoTime authors' existing recovery programme must be ruled out for any precise new claim.

**Proposed Idea:** none accepted. Do not invent an architecture before the identification/class audit or use a historical checkpoint's failure to criticize a corrected method.

**Falsification:** abandon a proposed gap if an existing method solves its exact information/compute-matched setting, if it relies on unavailable future data, if the generator makes the target unidentified, if its apparent novelty is a disclosed defect, or if the claimed causal guarantee is only finite-prior accuracy.

**Feasibility:** a single RTX5090 completed full-suite native inference/scoring in40.75s after data preparation, with~720MiB peak allocated CUDA memory. Exact data, checkpoint, source, configs, logs and predictions are retained privately. The next identification audit is mainly analytic and inexpensive numeric inference; a new long neural run is not yet justified.

**Next Experiment:** audit native hidden-confounder temporal mechanisms against existing identification theory, then compare likelihood/proxy recovery or an observational-equivalence witness on independent cases. The target and model-class assumptions must be explicit before any new method is trained. This is the next discriminator, not an accepted contribution.
''')
 papers=[
 ('Principled Offline RL in the Presence of Rich Exogenous Information',2023,'ICML/PMLR','https://proceedings.mlr.press/v202/islam23a.html','method + theory',4,4,4,'Primary paper assumptions and selected native continuous-action source read; multistep inverse already exists.'),
 ('No Gaussian Required: Contrastive Inverse Dynamics for JEPA World Models',2026,'arXiv2608.17542v1','https://arxiv.org/html/2608.17542v1','method + benchmark',3,4,4,'Primary objective/limitations and native scoring function read/executed; scope is anti-collapse, not an unconditional sufficiency theorem.'),
 ('Preserving Unstable Modes Through Inverse Dynamics in JEPA World Models',2026,'arXiv2610.07540v1','https://arxiv.org/html/2610.07540v1','method + theory',4,4,4,'Linear theorem explicitly states action-rank restrictions; selected native trainer/config audited, not trained.'),
 ('Learning Action-based Representations Using Invariance',2024,'arXiv2403.16369v2','https://arxiv.org/html/2403.16369v2','method + theory',4,3,3,'Recursive action-bisimulation and inverse-model limitations read; no execution.'),
 ('Guaranteed Discovery of Controllable Latent States with Multi-Step Inverse Models',2022,'Primary project page','https://controllable-latent-state.github.io/','method + theory',4,3,3,'Project-level screening; reinforces older multistep sufficiency boundary, no full proof audit claimed.'),
 ('Noise Contrastive Meta-Learning for Conditional Density Estimation using Kernel Mean Embeddings',2019,'arXiv1906.02236v2','https://arxiv.org/html/1906.02236v2','pure method',3,3,3,'Existing conditional kernel/density-ratio machinery; no novelty claimed for quadratic feature score.'),
 ('A Response Theory Probe for Learned Stochastic AI Simulators, Tested on Lorenz-63',2026,'arXiv2610.06798v1','https://arxiv.org/html/2610.06798v1','method + benchmark',4,4,4,'Primary method/results/limitations read; broad invariant-statistics versus forced-response question is already explicit. No native execution or official code availability asserted.'),
 ('Causal Identification in Time Series Models',2025,'CLeaR/PMLR','https://proceedings.mlr.press/v275/jahn25a.html','theory/proof',4,3,'N/A theory','Primary abstract screened: finite graph-segment identification results must be checked before claiming temporal identification is new.'),
 ('On the Identifiability of Controlled World Models',2026,'arXiv2607.22430v1','https://arxiv.org/html/2607.22430v1','theory/proof',4,4,3,'Primary assumptions/proof read: invertible observation, stationary linear Gaussian latent dynamics, policy-relative excitation. Not a theorem for arbitrary latent confounding.'),
 ('DoTime: A Synthetic Benchmark Generator for Interventional and Counterfactual Time Series',2026,'arXiv2607.27263 plus current official artifact','https://github.com/thummd/dotime/tree/8f227406a5b17c2040cfdcc680b493729ef47d00','method + benchmark',3,4,'N/A benchmark','Current repository errata supersede original advantage claim; native model/source/data actually executed here. Original paper acceptance not asserted.'),
 ('Correct then Forecast: Observer State-Space Models for Time Series Forecasting',2026,'arXiv2609.33566v1','https://arxiv.org/html/2609.33566v1','pure method',3,3,3,'Primary source screened: observer/propagator separation already proposed; supplementary code mentioned, no public standalone release verified.'),
 ]
 folder='literature-search-20261016-inverse-response'
 fields=['title','year','venue_or_source','link','paper_type','insight_score','completeness_score','numeric_evidence_score','relevance_note']
 path=ROOT/folder/'papers.csv';path.parent.mkdir(exist_ok=True)
 with path.open('w',newline='',encoding='utf-8') as f:
  writer=csv.writer(f);writer.writerow(fields);writer.writerows(papers)
 paperlines='\n'.join(f"- [{p[0]}]({p[3]}) — {p[2]}; {p[4]}; judgment I/C/N={p[5]}/{p[6]}/{p[7]}. {p[8]}" for p in papers)
 write(folder+'/papers.md','# Primary literature screening\n\nExploratory mode. Scores are internal screening judgments, not venue acceptance predictions. Depth is stated per entry.\n\n'+paperlines)
 write('literature_review.md','''# Cycle9 literature and novelty boundary

The detailed primary-source screening table is in `literature-search-20261016-inverse-response/papers.md` and `.csv`. Selected native source snapshots and complete tree inventories are retained;5primary PDFs are hash-archived, with no retrieval errors.

Multistep inverse learning is covered by ACRO/AC-State; recursive invariance by action-bisimulation. EP-IDM's preservation theorem is linear, assumes rich action support and explicitly acknowledges the endpoint rank constraint. The nonlinear loop diagnostic does not refute it. AC-MTM's native score uses raw-action squared distance; its paper distinguishes anti-collapse from task-relevant state coverage and reports a PushT deficit. Our new function-level test supplies a narrow witness, resolved by an existing generic score, not a new method.

The October5response-theory probe directly tests the separation between invariant statistics and forced response, including SINDy/RC/NODE/NSDE and calibrated truth controls. Its experiment is one low-dimensional system, one force/observable; those limits preserve specific extension opportunities, but repeating the broad separation would not be new.

DoTime's original preprint claim must be read with current official errata and re-test ledgers. The native1.2.0data and original frozen checkpoint were executed here; the later s13/feature-repair programme is existing prior work. Low s9 accuracy is not an unresolved novelty claim. Bayesian finite-prior prediction, graphical nonparametric identification and time-series parametric recovery must be distinguished in the next audit.
''')
 write(folder+'/search-notes.md','''# Search notes

Date:2026-10-16; purpose: exploratory direction discovery and falsification, not manuscript writing. Public-safe queries used multistep inverse dynamics, nonlinear controllability, conditional density contrast, time-series world models, response theory/chaotic emulators, latent-confounded temporal identification, and official DoTime checkpoint/erratum terms. Primary arXiv/PMLR/project/repository sources were verified. MDPI and preprints.org were excluded; secondary summaries, search snippets and explicitly synthetic/evaluation-only pages were not used as scientific evidence. Irrelevant search hits were discarded.

Important scope corrections: no unconditional inverse-sufficiency claim attributed to AC-MTM; EP-IDM explicitly recognizes rank/linearity restrictions; original DoTime interventional advantage has an erratum; s13 checkpoints are visible but not executed. A public file named prereg_2026-11 contains an October registration; its filename alone is not an execution date. No future result is inferred from it.

Raw GitHub-host retrieval timed out; the ended source-fetch process was replaced by TLS-verified GitHub blob API retrieval, preserving exact Git blob SHA1 and SHA256 checks. Only selected files were fetched; full trees are inventoried. No launcher, remote hardware or public write operation was performed. Literature PDF archive is primary preprint material; publication venue is not inferred.
''')
 write(folder+'/idea-grounding.md','''# Grounding packet

Direct observations: restricted raw-action distance NCE can ignore endpoint information carried by higher-order action statistics; ordinary quadratic contrast uses it. Original released DoTime s9 model has weak effect-sign accuracy on the complete corrected suite, but current author records already explain the broad failure.

Inference, low confidence: a useful next question may concern how a world model distinguishes effect information constrained by temporal data from prior-dependent predictions under latent confounding. This remains a question, not an established gap.

Already covered: multistep inverse representations, conditional density contrast, forecast-versus-forced-response audits, generic nonrectangular ambiguity, observer/dynamics separation, current DoTime recovery changes. Possible differentiation requires exact temporal identification assumptions, independent cases and strong existing likelihood/proxy/sensitivity controls. A standard method resolving the next test should end that local claim. No new method primitive is endorsed.
''')
 write('README.md','''# Autonomous time-series world-model discovery: cycle9

This is a completed evidence checkpoint; the full research goal remains active.

Server directory: `D:\\tswm_discovery_20261016_v1\\cycle9`. Original source dependency used during DoTime execution: `D:\\tswm_discovery_20261016_v1\\upstream\\dotime`, revision8f227406a5b17c2040cfdcc680b493729ef47d00. Byte-identical package/protocol files are retained under `sources/dotime` with a SHA256 manifest.

Use the existing project `.venv\\Scripts\\python.exe -B -u`. `scripts/runtime_profile.py` must precede numeric imports. Run `scripts/inverse_score_audit.py` for the new native-loss diagnostic. `scripts/evaluate_dotime_native.py` evaluates the fixed540episode subset; append `--full` for all12,150episodes. The DoTime evaluator expects the prior pinned upstream dependency and the cycle-local pfns vendor. The retained copy can restore that dependency if reproducing on another machine. No training or hyperparameter tuning is needed for the frozen model.

Checkpoint and released data are downloaded directly on the server, hash-checked and retained in `data`. Source/configs, metrics, per-episode predictions, raw scientific logs, primary PDFs and the7required research ledgers are included. Do not interpret historical native defects or a known generic repair as a novel contribution. Do not overwrite prior cycles or this checkpoint after sealing.
''')
 status={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'objective_achieved':False,'goal_status':'active','cycle9':{'scientific_jobs_complete':True,'inverse_fits':6,'native_checkpoint_inference_passes':2,'subset_episodes':540,'full_episodes':12150,'accepted_research_directions':0},'next_discriminator':'Exact temporal identifiability versus prior-dependent effect prediction; no new method or simulation cohort frozen','errors_resolved':['Ended raw GitHub fetch timeout bypassed with SHA-verified blob API','Dataset metadata-print attribute corrected after cached download; no re-download or training restart'],'scope_limits':['Native loss diagnostic is controlled transfer, not native visual RL reproduction','Only original published s9 causal checkpoint executed; corrected s13 and redesigned models not assessed','NaiveOLS predictive score is not an identification guarantee']}
 (ROOT/'outputs/task_status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
 print('Wrote7research ledgers, literature packet, registry and task status; full goal remains active')
if __name__=='__main__':main()
