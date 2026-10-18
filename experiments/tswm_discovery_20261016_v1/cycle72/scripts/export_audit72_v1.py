"""Close the input-contract audit with all fixed results and explicit negative evidence."""
import sys,pathlib,csv,io,math,base64
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common72_v3 import *
result=read(OUT/'FACTUAL_CONTROLS72_V3.json');identity=read(OUT/'BLOCK_IDENTITY72_V1.json')
assert result['status']=='FACTUAL_ONLY_DIAGNOSTICS_AND_TWO_NATIVE_KDE_FITS_CLOSED'
assert read(OUT/'factual_controls72_v3_exit.json')['exit_code']==0
assert not result['counterfactual_scoring_executed'] and not result['operational_forecasting_performance_claimed']
assert result['runtime']['all_native_tensors_cpu'] and not result['runtime']['cuda_available_observed'] and not result['runtime']['cuda_initialized']
assert read(OUT/'NATIVE_CONTRACT_QUALIFICATION72_V2.json')['single_counterfactual_states_directly_enter_loader']
selection=read(ROOT/'SOURCE_SELECTION72_V1.json');source=PROJECT/'upstream'/'epi-cf-benchmark'
for r in selection['files']:assert sha(source/r['path'])==r['sha256']
acq=read(OUT/'DATA_ACQUISITION72_V1.json')
for r in acq['files']+acq['additional_documentation']:assert sha(storage('data')/r['storage_relative'])==r['sha256']
records={s:read(OUT/('FACTUAL_CONTROLS_'+s.upper()+'72_V3.json')) for s in ('single','multi')}
for s,r in records.items():
    assert r['source_rows_removed']==0 and r['simulation_blocks']==158 and r['unique_county_ids']==153
    assert not r['counterfactual_target_or_compartment_values_parsed']
    assert len(r['per_block'])==158 and len(r['diagnostic_results'])==6
    model=storage('checkpoint')/('native_kde_'+s+'_v3')
    for a in r['native_kde']['artifacts']:assert sha(model/a['file'])==a['sha256']

compact=[]
for s,r in records.items():
    for arm in [a['arm'] for a in r['diagnostic_results']]:
        p=OUT/('factual_'+s+'_'+arm+'_per_block72_v1.csv')
        with p.open('x',encoding='utf-8',newline='') as f:
            w=csv.writer(f);w.writerow(['setting','instance','county_id','fit_days','diagnostic_days','mse','mae'])
            for row in r['per_block']:w.writerow([s,row['instance'],row['county'],row['fit_days'],row['diagnostic_days'],row['errors'][arm]['mse'],row['errors'][arm]['mae']])
        compact.append(p)
    native=OUT/('NATIVE_KDE_'+s.upper()+'_TRAINING72_V1.json')
    dump(native,dict(setting=s,**r['native_kde']));compact.append(native)
identity_summary=[]
for r in identity['settings']:
    p=OUT/('block_identity_'+r['setting']+'72_v1.csv')
    with p.open('x',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['instance','county_id','selected_factual_block_sha256','date_start','date_end'])
        for b in r['blocks']:w.writerow([b['instance'],b['county'],b['selected_factual_block_sha256'],b['date_start'],b['date_end']])
    compact.append(p);identity_summary.append({k:v for k,v in r.items() if k!='blocks'})
dump(OUT/'BLOCK_IDENTITY_SUMMARY72_V1.json',dict(utc=utc(),settings=identity_summary,source_rows_removed=0,counterfactual_values_parsed=False))
compact.append(OUT/'BLOCK_IDENTITY_SUMMARY72_V1.json')

table='| Fixed diagnostic | Single-policy MSE | Multi-policy MSE |\n|---|---:|---:|\n'
for a,b in zip(records['single']['diagnostic_results'],records['multi']['diagnostic_results']):
    assert a['arm']==b['arm'];table+='| '+a['arm']+' | '+format(a['raw_mse'],'.8g')+' | '+format(b['raw_mse'],'.8g')+' |\n'
lit='''# Literature and novelty boundary

[EpiCF-Bench, KDD2026](https://arxiv.org/html/2606.05692v3) supplies ABM-generated factual/counterfactual epidemic trajectories and evaluates dynamic policies. This is simulator truth grounded in empirical inputs, not observed human counterfactual truth. Its pinned public ML source is [8f7df275](https://github.com/complex-ai-lab/epi-cf-benchmark/tree/8f7df27507dc8b01422bb373417c8d0d3efffae2). C72's executed source contract, rather than paper-table inference, establishes that scenario-specific S/E/R/M are passed to conditional baselines. A history-and-plan-only world-model task must be compared in a separate information group.

[A Response Theory Probe,5October2026](https://arxiv.org/html/2610.06798v1) already separates invariant-statistics fidelity from mode-resolved forced response on stochastic Lorenz63, with actual SINDy/MLP/reservoir/NODE/NSDE experiments and independent true-realization calibration. Its correct-library sparse control performs well. Thus neither a generic accuracy-versus-response divergence nor a new Lorenz toy repeating it is a standalone new contribution. No reproduction of that paper is claimed here.

[G-Net](https://proceedings.mlr.press/v158/li21a/li21a.pdf) and [G-Transformer](https://arxiv.org/abs/2406.05504) already learn conditional covariate distributions and simulate trajectories for g-computation. Jointly generating future endogenous covariates is therefore an existing baseline principle, not a new architecture proposal. Their stated causal assumptions still require verification in any transferred benchmark. No G-Net/G-Transformer execution is claimed in C72.

The present branch yields an executable information-contract audit and bounded negative readout evidence. It does not establish a new world-model gap. Earlier C71 observation/interface/composition and continual-representation candidates remain provisional; this audit cannot substitute for their missing actual capability evidence.
'''
baseline='''# Baseline and data registry

EpiCF source: complex-ai-lab/epi-cf-benchmark8f7df27507dc8b01422bb373417c8d0d3efffae2, treef11d8c24963a226e227e6a5519e5e94435274f50.27 selected files/64085 bytes pass Git-blob and SHA256 checks. Three prior metadata/source files were reused,24 missing files added, historical metadata preserved. No license file appears in the inspected tree; third-party source is retained only on the server and is not redistributed in this publication.

Data: [Zenodo20680219,v1.1](https://doi.org/10.5281/zenodo.20680219), CC-BY4.0. Latest [20762823,v1.2](https://doi.org/10.5281/zenodo.20762823) has identical single/multi numeric CSV checksums; its dictionary changed. Both dictionaries are retained separately. Numeric data stay on serverG. Each setting contains26544 rows/158 contiguous168-day simulation blocks/153 distinct county IDs. Five repeated-ID pairs match exactly on the selected factual numeric fields and dates; no CF equality is assumed. All158 blocks are retained.

Executed: native preprocess_single/preprocess_multi, FluDataset, collect_x and flatten_features on constructed sentinels; two native KDE training fits on factual-only training exports; six fixed factual readout controls per setting. The KDE's in-sample bandwidth grid/selection is preserved; both choose0.1. There is no actual counterfactual inference, paper-table reproduction or operational forecast score. A separate grouping fix uses474 test instances for the three stacked multi-policy alternatives; its correctness was checked only on constructed data here.

Runtime: Python3.12 shared Torch2.11.0+cu128/NumPy2.4.3 retained; project-local pandas2.3.3 and pinned dependencies on serverG. CPU-only/two numerical threads and process-local core affinity; no shared package, system, driver or unrelated-process modification. All native training tensors were verified CPU; successful control process reports CUDA unavailable/uninitialized. C71's independent bounded campaign continues separately.
'''
experimental='''# Executed evidence and interpretation

Native source and dataset payload hashes are in SOURCE_AUDIT72_V1.json and DATA_ACQUISITION72_V1.json. The constructed native contract test changes only CF compartment columns, leaves target/TRAIN exports unchanged, and observes those changes reaching the native test loader and whole-sequence KDE features. It accesses no actual benchmark CF values.

The factual discriminator uses days1..111 for fitting and112..167 for checking each block:8848 diagnostic points/setting. All six arms are fixed before scores and are reported below in original case-count units. They receive contemporaneous factual compartment states; these are conditional observation-decoding errors, not future forecasts or counterfactual prediction errors.

'''+table+'''
Stock complement fails to beat the constant on pooled MSE in either setting. Linear level/FIR readouts have lower median block errors in some comparisons but substantially worse pooled errors; the full per-block errors are retained. The minus-S-change control improves some aggregate errors but is far from a near-exact observation decoder. This does not establish that no nonlinear/delayed decoder could work. No model/hyperparameter was selected from CF labels, and no additional decoder was tuned after these scores.

Two exact native KDE training functions complete on all158 factual blocks using native preprocessing helper outputs. Their normalization and bandwidth selection use the source's factual training protocol. Training losses are in-sample and cannot establish generalization or mechanism consistency. Model arrays stay on the server; compact training receipts and hashes are published.
'''
negative='''# Negative evidence and implementation limitations

- The predeclared algebraic/simple-linear shortcut did not nearly reconstruct the factual targets on the held-out temporal segment. The stock-complement arm is worse than the constant; generic linear arms have large aggregate errors. Do not claim the benchmark can be solved without dynamics from this experiment.
- An input contract containing scenario-specific future compartment states differs from history-and-plan-only simulation. This is a capability boundary, not proof that the original paper is invalid, that its target is a controlled direct effect, or that a new architecture is necessary.
- General forward simulation of unknown endogenous covariates is already covered by g-computation model families. A routine information-availability correction or reimplementation is insufficient novelty.
- Five duplicate-ID pairs match on the selected factual fields; do not count158 independent counties or assume duplicate seed/CF pairing. Original source blocks were retained, not silently filtered.
- The first qualification import failed because the shared runtime lacked pandas; a pinned project-local prefix fixed it without changing shared NumPy/Torch. The first factual adapter stopped on a CUDA-visibility assertion before data access; explicit CPU allocation/hiding was then checked. The second stopped on the incorrect158-unique-ID expectation before fitting. The preserved failures are implementation/identity issues, not model-performance deficits.
- C72 performs no CF scoring, probability-calibration reproduction, independent held-out county evaluation or new-method validation. It cannot complete the research-discovery goal.
'''
hypotheses='''# Competing explanations after the closed audit

| Hypothesis | Evidence/status | Next discriminator if reopened |
|---|---|---|
| Published ML inputs include scenario-specific endogenous states | Supported by pinned source plus executable sentinel trace | Audit any transferred model's actual inference-time information; keep conditional/origin-only groups separate. |
| Those states make the target a nearly trivial affine/conservation decode | Not supported by the six fixed factual controls | No post-score tuning here. A different decoder study would need a new development protocol and independent evidence. |
|158 simulation blocks are158 independent county IDs | Contradicted on pinned payloads:153 IDs and five factual duplicate pairs | Group by county ID for a future county-held-out experiment; do not invent stochastic pairing. |
| Joint endogenous-state rollout itself would be a new mechanism | Contradicted as a novelty claim by G-Net/G-Transformer | A practical residual after complete existing joint-generation controls would be needed. |
| A broad accuracy/statistics-versus-response finding is new | Already addressed by the October response-theory probe and existing world-model literature | Identify a distinct capability and actual residual rather than rerunning a known toy. |

No accepted new research hypothesis follows from this audit. Return to the unchanged C71 complete-model/free-simulation discriminator.
'''
ideas='''# Development routes remain provisional

A: observation-interface/dynamics reuse remains the current development route, requiring a distinct learning interaction and an independent structural split beyond complete observer/modular controls. C71 is still baseline competence preparation, not its novelty proof.

B: continual representation compatibility remains a fallback; alignment, freezing and replay are established. A residual surviving matched ordinary controls is missing.

C: information-contract-aware capability measurement is only an instrumentation route. C72 supports a conditional-versus-simulation distinction but does not show a near-trivial decoder wins, does not score actual CF outcomes and does not establish a new benchmark/method contribution. Do not force this source-specific correction into a time-series world-model paper. Reopen only with an independent practically relevant misclassification across competent models and explicit closest-work separation.

At most these three development lineages are retained; accepted final directions0. No further readout tuning on these same diagnostic windows is justified. Continue the already frozen complete-model assessment instead.
'''
final='''# No final research direction is established

**Research Question:** Can a reusable time-series world model learn faithful evolution beyond conditional decoding and beyond complete existing modular/state-estimation or g-computation methods? C72 audits the information contract; it does not establish the sought new gap.

**Evidence:** A native sentinel experiment confirms scenario-specific S/E/R/M enter the ML inputs. Verified data contain158 simulation blocks but153 IDs with five factual duplicate pairs. Two native KDE training paths close; six fixed readouts per setting do not produce a near-exact state-to-target shortcut. No actual CF prediction is scored.

**Existing Methods:** G-Net/G-Transformer already generate endogenous covariates through g-computation. Response-theory and current world-model work already separate common accuracy/statistics metrics from response fidelity. Original EpiCF source and conditional task are not declared invalid.

**Proposed Idea:** None accepted. Retain the observation/dynamics reuse development route and its existing fallback lineages; there is no distinct supported new mechanism here.

**Falsification:** The proposed simple decoder shortcut is not supported by its fixed controls. An ordinary joint-state rollout solving a later history-only task defeats novelty of that routine repair. Input/provenance mistakes or insufficient baseline training cannot be promoted to a new research gap.

**Feasibility:** C72's actual CPU runs and pinned small data fit the available server. C71's native relational training continues on the5090 with qualified bounded workers. Raw datasets/weights/arrays remain on serverG; selected code, hashes and reports go to GitHub.

**Next Experiment:** Finish the unchanged three-fit C71 free-simulation assessment, including ordinary history controls and separate privileged graph attribution. Do not add a new decoder sweep or score C72 CF labels to chase a positive result. A later causal-simulation experiment would need a separately frozen history/plan information contract and competent existing joint-generation baseline.

Accepted final directions0. Overall objective ACTIVE/UNACHIEVED. C72 is a closed audit, not final research completion.
'''
reports=dict(literature_review=lit,baseline_registry=baseline,experimental_results=experimental,negative_evidence=negative,hypothesis_space=hypotheses,research_ideas=ideas,final_research_direction=final)
for name,text in reports.items():
    with (ROOT/(name+'.md')).open('x',encoding='utf-8') as f:f.write(text)
procs=json.loads(os.environ['C72_LIVE_C71_JSON']);procs=procs if isinstance(procs,list) else [procs]
assert any('bounded_campaign71_v1.py formal' in r['CommandLine'] for r in procs)
assert any('bounded_native71_v1.py --request' in r['CommandLine'] for r in procs)
assert any('queue_free_simulation71_v3.py' in r['CommandLine'] for r in procs)
dump(OUT/'CLOSURE72_V1.json',dict(utc=utc(),status='INPUT_CONTRACT_AND_FACTUAL_CONTROL_AUDIT_CLOSED_NO_FINAL_DIRECTION',
    control_receipt_sha256=sha(OUT/'FACTUAL_CONTROLS72_V3.json'),native_exit=read(OUT/'factual_controls72_v3_exit.json'),
    source_sha256=sha(OUT/'SOURCE_AUDIT72_V1.json'),data_sha256=sha(OUT/'DATA_ACQUISITION72_V1.json'),
    compact_result_files=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in compact],
    report_hashes={n+'.md':sha(ROOT/(n+'.md')) for n in reports},live_c71_processes=procs,
    counterfactual_scoring_executed=False,new_method=False,accepted_final_directions=0,overall_goal='ACTIVE_UNACHIEVED',local_result_data_or_model_files_written=0))
status='''# C72 closed audit; native C71 training and research goal remain open

C72 confirms the native EpiCF ML input contract includes scenario-specific compartment states, which differs from history-and-plan-only simulation. The pinned data have158168-day blocks but153 distinct county IDs; five repeated-ID pairs agree on selected factual fields. The six fixed factual readout controls do not support a near-trivial decoder claim. Two native KDE training paths close, but training reconstruction is not held-out forecasting or causal simulation. No actual CF labels/states were semantically used or scored. See cycle72/experimental_results.md and negative_evidence.md for all fixed results and limits.

The source/input/identity fixes are not an accepted new world-model contribution. G-computation families already model future endogenous covariates, and recent response-theory work already studies accuracy/statistics-versus-response divergences. No final direction accepted; full goal ACTIVE/UNACHIEVED. The independent C71 bounded controller, native worker and frozen assessment queue were verified live when CLOSURE72_V1.json was written. They remain incomplete and require current process/terminal-receipt inspection. Data/models/arrays stay on the authorized server storage; GitHub receives authored code and compact evidence. Original third-party source is not redistributed.
'''
with (PROJECT/'CURRENT_RESEARCH_STATUS_V72.md').open('x',encoding='utf-8') as f:f.write(status)
closed=['CLOSURE72_V1.json','SOURCE_AUDIT72_V1.json','SOURCE_TRANSPORT72_V1.json','METADATA_ACQUISITION72_V1.json','DATA_ACQUISITION72_V1.json','DEPENDENCIES72_V1.json','NATIVE_QUALIFICATION_ADAPTER72_V2.json','NATIVE_CONTRACT_QUALIFICATION72_V2.json','FACTUAL_MISSINGNESS72_V1.json','FACTUAL_CONTROLS_ADAPTER72_V2.json','FACTUAL_CONTROLS72_V3.json','factual_controls72_v1_exit.json','factual_controls72_v1_stdout.log','factual_controls72_v2_exit.json','factual_controls72_v2_stdout.log','factual_controls72_v3_launch.json','factual_controls72_v3_exit.json','factual_controls72_v3_stdout.log','native_kde_single_v3_stdout.log','native_kde_multi_v3_stdout.log']
files=[p for p in ROOT.iterdir() if p.is_file() and p.suffix in ('.md','.json','.ps1')]+list((ROOT/'scripts').glob('*.py'))+[OUT/n for n in closed]+compact
dest=OUT/'publication_v1';dest.mkdir(exist_ok=False);patterns=json.loads(os.environ['C72_PRIVATE_EXPORT_PATTERNS_JSON']);rows=[]
for p in sorted(set(files)):
    raw=p.read_bytes();rel=p.relative_to(ROOT).as_posix()
    for pattern in patterns:assert pattern.encode() not in raw,(rel,'private export pattern')
    try:body=raw.decode('utf-8-sig');encoding='UTF-8'
    except UnicodeDecodeError:body=base64.b64encode(raw).decode('ascii');rel+='.base64.txt';encoding='Base64 exact original bytes'
    b=body.encode('utf-8');target=dest/'files'/rel;target.parent.mkdir(exist_ok=True,parents=True);target.write_bytes(b)
    rows.append(dict(path=rel,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),git_blob_sha1=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest(),encoding=encoding,server_snapshot_file=str(target),parts=math.ceil(len(body)/3500)))
manifest=dict(utc=utc(),status='C72_AUDIT_CLOSED_NO_FINAL_DIRECTION',files=rows,file_count=len(rows),bytes=sum(r['bytes'] for r in rows),original_upstream_code_raw_data_weights_and_arrays_included=False,
    counterfactual_scoring_executed=False,accepted_final_directions=0,overall_goal='ACTIVE_UNACHIEVED',local_result_data_or_model_files_written=0)
dump(dest/'manifest.json',manifest);print(json.dumps(dict(status=manifest['status'],files=len(rows),bytes=manifest['bytes'],manifest_sha256=sha(dest/'manifest.json'))),flush=True)
