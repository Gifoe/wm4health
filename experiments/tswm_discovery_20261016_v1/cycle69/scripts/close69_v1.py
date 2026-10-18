import sys,pathlib,platform,importlib.metadata,subprocess
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common69_v1 import *
assert read(OUT/'campaign69_v1_exit.json')['exit_code']==read(OUT/'replay69_v1_exit.json')['exit_code']==read(OUT/'diagnose69_v1_exit.json')['exit_code']==0
replay=read(OUT/'INDEPENDENT_REPLAY69_V1.json');assert replay['status']=='PASS';r=read(OUT/'RESULTS69_V1.json');ld=read(OUT/'FORECAST_LEDGER69_V1.json');d=read(OUT/'WH_DIAGNOSTIC69_V1.json');q=read(OUT/'QUALIFICATION69_V1.json');cm=read(OUT/'DATA_MANIFEST69_V1.json');context=read(ROOT/'SOURCE_CONTEXT69_V1.json');assert d['primary_results_unchanged_sha256']==sha(OUT/'RESULTS69_V1.json')
lookup={(x['record'],x['method']):x for x in r['rows']};diag={x['mode']:x for x in d['rows']};wh=lookup[('WH0_initial50','meta_precision')]['RMSE'];paper=context['repository_pdf']['reported_RMSE'];assert len(r['rows'])==28 and len(ld['records'])==7
analysis=dict(utc=utc(),status='REAL_BENCHMARK_EXECUTION_AND_DIAGNOSTIC_EVIDENCE_CLOSED',reproduction_status='CED_CLOSE_TANKS_NEAR_WH_DISCREPANCY_UNRESOLVED',physical_systems=3,trajectories=4,protocol_records=7,primary_rows=28,primary_scored_point_records=replay['counts']['scored_point_records'],native_meta_parameters=q['native_meta_parameters'],released_dyno_parameters_per_checkpoint=323,distinct_neural_checkpoints=4,neural_training_this_cycle=0,primary_WH_RMSE=wh,paper_reported_WH_RMSE=paper['WH'],WH_post_discovery_diagnostics=diag,primary_results_not_replaced=True,negative_evidence=['Generic Bayesian context aggregation has prior art and earlier project controls.','WH discrepancy is present in the first query; late-horizon drift is not its dominant explanation.','Fused feedback and even privileged measured-boundary resets leave most WH error.','Changing the scored tail to notebook75000 points does not explain the discrepancy.','Code architecture did not change between the weight release and the pinned source.','Checkpoint-to-paper-table mapping and exact publication protocol remain unverified; do not infer a new architecture requirement.'],accepted_final_directions=0,overall_goal='ACTIVE_UNACHIEVED')
dump(OUT/'ANALYSIS69_V1.json',analysis)
def write(name,text):
 with (ROOT/name).open('x',encoding='utf-8') as f:f.write(text+'\n')
write('README.md','''# C69: original physical-system task family, native execution

Three public physical systems/four trajectories, seven initialization records and28 whole-trajectory metric rows executed. Independent replay passed324460 scored point records, all official splits, ensemble moments and fixed fresh native queries. One full pretrained Meta-ID checkpoint and three323-parameter target-trained author dynoNet artifacts were used; zero neural fits occurred here.

Reproduction is partial: CED scores are close to the paper, Tanks near its reported range, WH remains discrepant. Read INTERPRETATION69_V1.md for why this is not a new world-model research gap. Original primary results are preserved; two post-discovery WH diagnostics are separately labeled, including a privileged control. Data/weights/forecast arrays stay on server G; no local result files or third-party payloads are published. Overall research goal ACTIVE / UNACHIEVED; accepted directions0.
''')
write('literature_review.md',(ROOT/'LITERATURE_DELTA69_V1.md').read_text(encoding='utf-8'))
write('baseline_registry.md',f'''# Baseline identity and execution registry

| Artifact | Actual execution | Information/provenance boundary |
|---|---|---|
| Meta-ID ckpt_430_fine-tuned_3100000_lower_noise | Complete{q['native_meta_parameters']} parameters /204 state tensors, frozen | Synthetic pretraining; training records used as context only. Exact source and cached/uncached equivalence verified. Original meta training not reproduced. |
| Cached TSTransformerBenchmark | Same weights and source computation | Native cached class, not a proxy model. Source encoder output precomputed; clear cache per task/protocol. |
| dynoNet WH author artifact |323 parameters /10 state tensors, epoch8000 | Target-trained released checkpoint; selection/training protocol not independently reconstructed. |
| dynoNet Tanks author artifact |323 parameters /10 state tensors, epoch8000 | Separate target-trained checkpoint. |
| dynoNet CED author artifact |323 parameters /10 state tensors, epoch8000 | Same checkpoint for both CED trajectories; scaler uses both training records, unlike per-trajectory meta context. |
| Context mean | Executed from training values | Sanity baseline without known future inputs. |

No published result is copied into our experimental rows. Paper figures are only comparison references. Whole pretrained weights are strict-loaded, key/value-checked and SHA-verified. Only exact selected source AST definitions execute for the official loader and dynoNet adapter; downloader, TLS fallback, argparse launchers and cache deletion never execute. Sources/protocols are pinned in SOURCE_SELECTION69_V1.json, DATA_SOURCE69_V2.json and SOURCE_CONTEXT69_V1.json.
''')
lines=['| Record / initialization | Method | RMSE | MAE | common-suffix RMSE | scored points |','|---|---|---:|---:|---:|---:|']
for x in r['rows']:lines.append('|'+x['record']+'|'+x['method']+'|'+ '|'.join(f'{x[k]:.9g}' for k in ['RMSE','MAE','common_suffix_RMSE'])+'|'+str(x['scored_points'])+'|')
write('experimental_results.md','''# Primary complete-trajectory results

Units are the released signal units. Do not average these systems into one raw-unit leaderboard or treat contexts/time points as independent systems. WH input/output splits follow exact official loader[5200:184000], then100000 training points and78800 test points. CED splits each500-point trajectory at400. Tanks uses its provided train/test arrays. All numeric files and split values independently verify.

The two preregistered information contracts are retained: Tanks5 versus50 initial outputs; CED4 versus10. WH uses50 in both. Only the allowed initial test outputs enter full simulation; later query prefixes are each expert's own predictions. Every remaining target is scored, including the final partial100-point block. Context windows are250 disjoint400-point windows for WH,61 overlapping windows for Tanks and1 for each CED trajectory. Float32 training-only source normalization; float64 ensemble fusion/scoring. No test hyperparameter selection.

'''+ '\n'.join(lines)+'''

dynoNet artifacts use additional target-domain training; this is a separate information group. CED dynoNet's scaler uses both training trajectories. The source ignores measured initial y and simulates from its native internal initialization; retain that behavior. The three released artifact training/selection histories were not reconstructed. Thus the table is not an equal-pretraining or equal-training-budget method ranking.

Per-context means/stds and every point forecast remain in hashed server G archives. RESULTS69_V1.json also provides early/late halves and first/last100 errors. Estimator and mixture-moment3SD fractions are descriptive: the former assumes independent expert errors and the latter is a marginal moment approximation, neither a validated trajectory probability law. No calibration parameter is fitted to test.
''')
write('INTERPRETATION69_V1.md',f'''# What changes after the physical-system audit

Existing contextual simulation is a useful strong neighbor, not merely an abstract prior-art objection. CED paper-prefix RMSEs are {lookup[('CED0_initial4','meta_precision')]['RMSE']:.6f} and {lookup[('CED1_initial4','meta_precision')]['RMSE']:.6f}; the paper's reference values are.068/.055. Tanks paper-prefix precision fusion yields {lookup[('Tanks0_initial5','meta_precision')]['RMSE']:.6f} versus the reference.429. These close/near results support retaining this fully executed baseline. They are not proof of the publication's exact full protocol.

WH remains unreplicated: our full-tail precision RMSE {wh:.8f}, versus reported.0014. Independent source-to-server data/weight hashes, manual splits, ensemble/scoring arithmetic, cached/uncached equality and fresh native calls pass. Repository PDF page11 has the same.0014, and the release-to-head comparison changes only README/PDF, not model code. The loader's last pre-release commit already has the same WH split and official initial lengths. The exact checkpoint/protocol used for the published table remains unverified; do not accuse the table of error or reinterpret this as an established field-wide research gap.

The discrepancy is not mainly a distant-rollout failure in this execution. First100 RMSE is {lookup[('WH0_initial50','meta_precision')]['first100_RMSE']:.8f}; early/late halves are {lookup[('WH0_initial50','meta_precision')]['early_half_RMSE']:.8f}/{lookup[('WH0_initial50','meta_precision')]['late_half_RMSE']:.8f}. Cropping to the notebook's75000 scored points gives {diag['primary_first75000']['RMSE']:.8f}; fused feedback gives {diag['fused_feedback']['RMSE']:.8f}; even privileged true-history resets every100 steps give {diag['privileged_measured_boundary']['RMSE']:.8f}. The last two are post-discovery explanatory controls, with independently checked metrics/fresh queries, not selected replacements for primary scores. A generic long-horizon recurrence fix lacks support here.

The released target-trained323-parameter dynoNet artifact reaches WH RMSE {lookup[('WH0_initial50','dyno_author')]['RMSE']:.8f}. This shows a familiar model can describe this physical task with enough target-domain data, while its training provenance prevents an equal-information superiority claim. It does not prove arbitrary world modeling solved.

The correct next action is not to tune this WH test until a positive result appears. Keep source discrepancies explicit, retain competent adjacent baselines and pursue a distinct reusable capability only after checking the closest structured/recognition-based state-space methods. No new final direction accepted. Goal ACTIVE / UNACHIEVED.
''')
write('negative_evidence.md','''# Negative evidence and non-scientific failures

- Generic context fusion/posterior repair already has Bayesian aggregation and structured-NP prior art, plus earlier project controls. No new framework follows from adding weights to windows.
- CED/Tanks do not support a blanket claim that a synthetic-trained contextual model fails on physical systems.
- WH error is already large in the first query and does not systematically grow across halves. Simple fused feedback and a privileged measured-boundary reset leave most error. Do not reuse a long-horizon drift story.
- Whole-tail versus75000-point cropping does not explain WH. Code architecture did not change from the release to pinned source; the repository PDF shares the reported reference. Remaining checkpoint/protocol correspondence is not established.
- Extremely narrow independent-expert estimator intervals are not automatically joint predictive intervals. Context expert errors may be correlated, especially for overlapping windows. This is a known statistical distinction, not a novel world-model problem by itself.

Preserved execution failures: initial numeric staging rejected CED's valid legacy MAT header, corrected by direct format-aware parsing; original WH host refused connection, so the official page's mirror provided the identical named MAT member with archive CRC. First qualification exited0xC0000005 during PyTorch import, before model loading; its handle was terminal, logs retained, then identical code/environment passed. No training, environment/driver modification or unrelated process termination occurred. These are not scientific failures of the models.
''')
write('hypothesis_space.md','''# Competing hypotheses after C69

| Hypothesis | Direct test | Status |
|---|---|---|
| More context aggregation itself is a new mechanism | Prior-art and earlier C25 controls | Rejected as a standalone novelty claim. |
| The actual pretrained contextual simulator has no useful physical transfer | Original-task-family native execution on3 systems | CED/Tanks near references contradict the blanket claim; WH remains discrepant. |
| WH is principally long-horizon accumulated error | First100/halves, fused feedback, privileged boundary resets | Not supported as the dominant explanation in this execution. |
| Scoring the incomplete final tail caused the WH mismatch | Saved primary output first75000 points | Not supported. |
| Post-release model-code drift explains the mismatch | Exact release/current comparison | Not supported over those commits; other protocol/checkpoint mapping is not established. |
| Higher inverse-variance fusion precision establishes predictive calibration | Interval diagnostics and correlation assumptions | Invalid implication; means/estimator variance are distinct from a joint predictive law. |
| A new framework has been validated | No new method, no clean confirmatory capability test | No. |

Keep target-trained artifacts, synthetic-pretrained contextual inference and privileged diagnostics in their own information groups. No test-selected method, seed, context length or calibration parameter.
''')
write('research_ideas.md','''# Candidate directions

No final direction accepted. Generic context aggregation and long-horizon feedback repair are not supported as new leads. A paper-specific reproduction discrepancy is not a research gap.

One provisional investigation, not a recommendation: determine the useful capability boundary between context-level system identification and online latent-state inference when both system parameters and hidden state are uncertain. Before any new method, read the complete RP-GSSM controlled-dynamics implementation alongside Meta-iLaD and existing HiP-RSSM/structured inference evidence. The project already has strong Bayesian/context and covariance controls; ordinary dual estimation, a Gaussian posterior or an observer cannot be renamed into novelty. A candidate would need a specific remaining failure and a distinct practical mechanism validated on independent systems. This audit has not shown that failure.
''')
write('final_research_direction.md','''# No accepted final direction

**Research Question:** Which reusable state/dynamics capability remains unresolved after strong contextual system identification and structured state-space inference? C69 has not established a novel answer.

**Evidence:** Exact source/weights executed on3 real physical systems/four trajectories,7 initialization records,28 whole-trajectory rows and324460 scored point records. All splits/ensembles/metrics replay; fixed fresh native calls agree. CED/Tanks transfer is useful, WH remains unreplicated despite named diagnostic exclusions.

**Existing Methods:** Synthetic-pretrained contextual simulation, dynoNet, Bayesian context aggregation and structured state-space inference already cover many ordinary repairs. Information and training budgets differ; neither small classical success nor an unresolved reproduction mismatch settles general world modeling.

**Proposed Idea:** None accepted. Do not turn uncertainty-fusion arithmetic or source/protocol sensitivity into a new architecture.

**Falsification:** Ordinary contextual/structured inference removing a candidate residual defeats a new-mechanism necessity claim. A gain requiring test outputs, extra target-domain labels or test hyperparameter selection does not count. An unreproduced paper score cannot establish a novel failure mechanism.

**Feasibility:** One RTX5090 executes the complete contextual model; CPU executes released323-parameter dynoNet artifacts. No neural training this cycle. Server G keeps data/weights/raw outputs; GitHub gets authored source/reports/compact evidence only.

**Next Experiment:** First inspect and qualify the actual controlled-dynamics example in RP-GSSM's author code, including its recognition factors, input-conditioned transition and any environment/state uncertainty assumptions. Compare that supported contract to Meta-iLaD and existing project covariance/context controls. Only select a new independent mechanism test if a distinct useful capability remains untested; otherwise reject that lead. This is the next concrete source/qualification action, not an accepted new problem or a completed experiment. Stop tuning WH benchmark settings.

Overall research goal ACTIVE / UNACHIEVED; accepted final directions0.
''')
write('REPRODUCE69_V1.md','''# Replay and reproduction scope

Keep sibling cycle68/scripts/native68_v1.py and cycle66/scripts/native66_v1.py accessible, with the unchanged C68 Meta-ID source/weight receipt. Configure project STORAGE_LOCATION_V1.json to put data/checkpoint/artifact roots in authorized server G storage. The pinned public files and original numeric URLs are in SOURCE_SELECTION69_V1.json. DATA_SOURCE69_V2.json records official numeric ZIP-member identities/CRC, source-to-server hashes and the two public transport fallbacks. No third-party payload is redistributed.

Use the recorded isolated salt_env versions, with its Windows Library/bin path. prepare69_v1.py executes only exact official loader/container AST definitions on acquired local MAT files. qualify69_v1.py strictly checks all model states, cached/uncached equality and hidden-future invariance. campaign69_v1.py runs all fixed complete-trajectory contracts, then replay69_v1.py manually reconstructs source splits and every metric/ensemble and repeats fixed native queries. diagnose69_v1.py follows a separate post-discovery protocol; it does not overwrite primary scores. Always use a fresh reproduction output/storage folder; writes are exclusive.

Original upstream training and checkpoint selection were not replayed. Our paper-prefix/full-tail and current-loader protocols are explicitly separate. Thus verifier PASS certifies this executable evidence, not complete exact reproduction of all paper tables. CED is close, Tanks near, WH unresolved. All original failed launch logs and the successful retry remain in outputs.
''')
with (PROJECT/'CURRENT_RESEARCH_STATUS_V69.md').open('x',encoding='utf-8') as f:f.write(f'''# Current status: C69

Real physical-system contextual simulation and independent replay closed:3 systems,4 trajectories,7 initialization records,28 rows/324460 scored point records; no new neural fits. CED/Tanks close or near references; WH RMSE {wh:.8f} versus reference.0014 remains unresolved. Ordinary fused-feedback, privileged boundary reset and final-tail crop do not explain most discrepancy. Code architecture unchanged across weight release/current head.

No accepted new framework. Goal ACTIVE / UNACHIEVED. Next concrete action: qualify RP-GSSM's actual controlled-dynamics code and inspect its state/environment uncertainty contract before selecting an independent capability test. Do not tune WH test settings. See cycle69/INTERPRETATION69_V1.md. Raw files remain server G; no local result synchronization.
''')
reports={p.name:sha(p) for p in ROOT.glob('*.md')};names=['DATA_SOURCE69_V2.json','DATA_MANIFEST69_V1.json','QUALIFICATION69_V1.json','RESULTS69_V1.json','FORECAST_LEDGER69_V1.json','INDEPENDENT_REPLAY69_V1.json','WH_DIAGNOSTIC69_V1.json','ANALYSIS69_V1.json','campaign69_v1_exit.json','replay69_v1_exit.json','diagnose69_v1_exit.json'];artifacts={n:sha(OUT/n) for n in names};sources={p.relative_to(ROOT).as_posix():sha(p) for p in list((ROOT/'scripts').glob('*.py'))+list(ROOT.glob('*.ps1'))+list(ROOT.glob('*.json'))};gpu=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
dump(OUT/'CLOSURE69_V1.json',dict(utc=utc(),status='CLOSED_NATIVE_REAL_BENCHMARK_PARTIAL_REPRODUCTION_AND_DIAGNOSTICS',reports=reports,artifacts=artifacts,sources=sources,environment=dict(Python=platform.python_version(),isolated=bool(sys.flags.isolated),executable=sys.executable,packages={x:importlib.metadata.version(x) for x in ['torch','numpy','scipy','scikit-learn','safetensors']},GPU=gpu,configured_TF32=False),neural_checkpoints=4,neural_fits=0,physical_systems=3,trajectories=4,primary_rows=28,scored_point_records=324460,independent_replay='PASS',paper_reproduction='PARTIAL_WH_UNRESOLVED',post_discovery_diagnostics='Separately declared, include privileged information, not selected replacement primary results',accepted_final_directions=0,overall_goal='ACTIVE_UNACHIEVED',local_result_files_written=0,third_party_payloads_published=False))
print(json.dumps(dict(status='CLOSED_NATIVE_REAL_BENCHMARK_PARTIAL_REPRODUCTION_AND_DIAGNOSTICS',closure_sha256=sha(OUT/'CLOSURE69_V1.json'),reports=len(reports),primary_rows=28,accepted_final_directions=0)),flush=True)
