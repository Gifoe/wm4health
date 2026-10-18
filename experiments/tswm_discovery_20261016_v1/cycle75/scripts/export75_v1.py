"""Publish complete compact evidence, retaining arrays/models only on server G."""
import sys,pathlib,csv,io,os,hashlib
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common75_v1 import *
result=read(OUT/'CALIBRATION_RESULTS75_V1.json');verification=read(OUT/'VERIFICATION75_V1.json');terminal=read(OUT/'run75_v1_exit.json')
assert terminal['exit_code']==0 and verification['result_sha256']==sha(OUT/'CALIBRATION_RESULTS75_V1.json')
assert len(result['rows'])==54 and verification['replay_checks']==486 and verification['forecast_values_verified']==2011392
assert all(r['failed_records']==0 for r in result['rows'])
values={(r['dataset'],r['condition'],r['method']):r['nrmse'] for r in result['rows']}
methods=['persistence','source_unadapted','privileged_known_sensor','source_affine_inverse','source_monotone_inverse','calibration_arx64']+['calibration_narx64_seed'+str(s) for s in [2026751,2026752,2026753]]
table='| Method | EMPS identity | EMPS affine | EMPS nonlinear | F16 identity | F16 affine | F16 nonlinear |\n|---|---:|---:|---:|---:|---:|---:|\n'
for method in methods:table+='| '+method+' | '+' | '.join(format(values[d,c,method],'.9f') for d in ['EMPS','F16'] for c in ['identity','affine','nonlinear'])+' |\n'
neural={}
for d in ['EMPS','F16']:
 for c in ['identity','affine','nonlinear']:
  vv=[values[d,c,'calibration_narx64_seed'+str(s)] for s in [2026751,2026752,2026753]];neural[d+'_'+c]=dict(mean=sum(vv)/3,min=min(vv),max=max(vv),values=vv)
report='''# C75: ordinary calibration/refit under unpaired constructed observation changes

No final research direction is accepted. This experiment weakens the premise that a new world-model mechanism is needed merely because an unadapted predictor fails after a sensor change. Ordinary calibration and refitting recover substantial predictive quality. The frozen one-step calibration remains imperfect on F16; that residual is not yet attributed to a new mechanism failure.

## Task and provenance

This is a newly designed diagnostic using nine real EMPS/F16 original TRAIN records from C74. The declared observation functions are artificial, invertible scalar transformations in each system's source-standardized coordinates: identity;0.35+1.4*y;0.25+1.2*y+0.8*tanh(1.5*y). These are not actual replaced sensors, joint changes in physical dynamics, or a source-paper reproduction. Physical units and sampling rates remain separate across systems. EMPS's recorded feedback motor force is supplied as a future input condition; no new-policy causal effect is estimated. F16 is the official default single acceleration channel, not the whole SIMO system.

Source is the first40% of each record. Calibration is the next40%, with its own first80% for fitting and final20% for internal selection. DEV is the original final20%, using the same C74 complete origins after its256-sample gap. Source-only pooled system scaling and two source ARX64 models are newly fitted: no C74 first80% coefficients/scaling are reused. Source and calibration trajectories are disjoint. Learners receive observed calibration(u,x) only, with no paired canonical calibration y or sensor generator. The known-sensor row is a privileged diagnostic. All methods receive128 observed history samples and the same supplied128 future inputs. Every future output is recursive, without target resets.

Ordinary inverse identification uses positive-slope affine or17-knot piecewise-linear functions. Its objective is a frozen-ARX innovation equation. Four fixed identity-prior weights are selected only by recursive observed-x error on internal calibration validation, not DEV. All48 candidate fits and scores are retained. Bounds are never active in any selected adapter. Its first beta parameter is the canonical value at the first knot, not a free physical offset at x=0. A returned map is not evidence of unique physical-state identification.

Calibration-data refits comprise ARX64 and three33,281-parameter residual NARX64 MLPs per condition/system. Each neural seed runs the full2000 one-step plus500 recursive16-step updates; checkpoints are chosen using only internal128-step calibration validation. There are18 neural fits and45,000 actual neural updates in total, plus two source and six calibration ridge fits. No test-based tuning, prediction clipping, failed-record filtering, or newly proposed architecture is used. Adapter arms additionally have their fitted source model; no equal-compute, sample-efficiency or capacity-superiority claim is made.

## Complete DEV results

NRMSE is divided by pooled observed-x SD from calibration fitting only. Systems and observation conditions are not pooled. All54 method/system/condition rows and all243 record rows are retained. These are repeated windows, not independent physical-system replications.

'''+table+'''
All three neural seeds are shown; no best-seed claim or significance test is made. Aggregate and record-level raw RMSE/MAE, selected checkpoints, all internal checkpoint scores, all inverse-map candidates and artifact hashes accompany this report. The privileged known-sensor source model is not a universal lower bound: a refit can improve its imperfect source dynamics.

On EMPS nonlinear observation, frozen unadapted NRMSE is0.040125129, ordinary monotone calibration0.023289393, calibration ARX0.016905994 and the three nonlinear refits0.010562858/0.014706616/0.020304213. Thus this setting does not support a necessity claim for a new latent reuse mechanism.

On F16 nonlinear observation, frozen unadapted NRMSE is0.521722905, ordinary monotone calibration0.409961400 and the three nonlinear refits0.349425181/0.327292385/0.342957464. The privileged source-model diagnostic is0.299313628. Frozen one-step calibration is inferior to ordinary nonlinear refitting here; this could be an objective/noise/model-bias issue. No simulation-error-trained sensor adapter, uncertainty-aware state estimator, full nonlinear state-space/replay control or genuinely changed physical dynamics is evaluated by this cycle. These omissions prevent promotion of the residual to a general world-model research gap.

## Independent qualification

Every saved target reconstructs exactly from hash-verified C74 canonical arrays and the C75 transformation. All2,011,392 forecast values and all aggregate/record metrics reconstruct exactly. An independent scalar recurrence and NumPy Linear/SiLU implementation replay486 first/last-window forecasts across every record, condition and method. Maximum float64 disagreement is1.7763568394002505e-14 and maximum Torch-float32 versus NumPy disagreement2.1330354433857224e-6. Inverse-map innovation equations and internal-only lambda selection also reconstruct. No learner forecast helper is reused in this replay.

The EMPS source AR characteristic radius is1.0001247463 and its offset innovation multiplier1-sum(a)=-0.00061866265. It is not a stable infinite-horizon physical simulator, and the near-unit offset sensitivity requires care in interpreting calibration coordinates. F16's radius is0.9933500055 and multiplier0.3200745115. The finite128-step task is reported as specified; no root stabilization was applied after seeing DEV. Source-array, coefficient, model, prediction, protocol and raw-log hashes remain traceable. The history-only API is an information boundary, not a future-target sentinel experiment.

## Prior art and decision

Unknown static observation nonlinearities around learned dynamics are an established system-identification setting. The primary [Bai and Reyland Automatica paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC3587721/) analyzes assumptions, scaling ambiguity and identification under nonlinear outputs. The [Uppsala Wiener identification project](https://www.it.uu.se/research/syscon/Ident/Wiener_Id.html) discusses nonlinear measurements, blind adaptation and validation. Their guarantees are not asserted to apply to these feedback/colored-input real records. This cycle is an ordinary baseline instantiation, not a faithful reproduction of either algorithm.

The primary [ReDRAW L4DC2026 abstract](https://proceedings.mlr.press/v331/lanier26a.html) reports latent-dynamics residual adaptation for sim-to-real visual world models. Only its official abstract/bibliography was inspected here; no code was executed or full theorem checked. It blocks a generic claim that latent dynamics adaptation is new, but does not prove this exact sequential unpaired sensor/dynamics reuse task solved. Earlier SplitJEPA, JEPA-TTT and compositional-continual boundaries remain in C71/NOVELTY_BOUNDARY71_V4.md. No absence-of-results inference is treated as positive novelty evidence.

The next discriminating control, before increasing architecture complexity, is an ordinary sensor adapter trained by multi-step simulation error on the same calibration records, with state initialization and noise treatment qualified. Compare it with the existing inverse-innovation baseline and nonlinear refits under frozen information and budgets. If it closes the F16 residual, reject the need for a new mechanism on this setting. If not, separate approximation, excitation, calibration ambiguity and measurement noise before constructing a sequential joint-change test. This is a next-control decision, not a final recommended paper direction.

Official test MAT outputs remain numerically closed. The independent C71 original-budget three-seed campaign is unchanged. All arrays/weights/raw data remain on configured server G storage; only selected authored code, compact evidence and reports are published. Overall goal ACTIVE/UNACHIEVED; accepted final directions0.
'''
with (ROOT/'CALIBRATION_FINDING75_V1.md').open('x',encoding='utf-8') as f:f.write(report)
entries={
 'literature_review.md':'Wiener nonlinear-output identification and latent dynamics residual adaptation are existing methods, not new ideas. See primary sources and scope qualifications in CALIBRATION_FINDING75_V1.md and the earlier C71 novelty boundary. No proof or empirical guarantee is transferred from their assumptions to these real feedback/colored-input records.',
 'baseline_registry.md':'Executed two new source ARX64 fits, six calibration ARX64 refits,48 inverse-map candidates and18 residual NARX64 fits(three seeds per condition/system;45,000 updates). Persistence and privileged known-sensor controls retained. Neural parameters33,281 each. No proposed model, original-paper reproduction or equal-budget superiority claim.',
 'experimental_results.md':'All54 aggregate and243 record rows are in outputs/aggregate_metrics75_v1.csv and outputs/record_metrics75_v1.csv. The full table follows.\n\n'+table+'\nAll2,011,392 forecast values, targets and metrics independently verified;486 independent recurrence checks. All records finite. Raw arrays/weights remain server-only.',
 'negative_evidence.md':'Ordinary calibration/refitting strongly reduces constructed sensor-shift error. EMPS nonlinear monotone adapter0.023289393 versus unadapted0.040125129; NARX seeds0.010562858/0.014706616/0.020304213. F16 nonlinear refits0.349425181/0.327292385/0.342957464 versus unadapted0.521722905. These reject using an unadapted predictor alone as evidence that a new world-model mechanism is necessary. F16 one-step-adapter residual is unresolved, not proof of novelty. No joint physical dynamics change or forgetting was tested.',
 'hypothesis_space.md':'H1 unadapted scalar observation change necessarily needs a new framework: not supported after ordinary controls. H2 one-step inverse innovation calibration is sufficient across both real systems: not established, especially F16. H3 the residual reflects objective/noise/source-model mismatch: open; test simulation-error-trained ordinary adapter before architecture design. H4 general unpaired sequential observation/dynamics reuse fails after all complete controls: untested here. Exact canonical coordinates need not be uniquely identifiable for predictive adequacy.',
 'research_ideas.md':'Do not promote sensor calibration itself, the real-data wrapper or the residual NARX baseline as an idea. Sequential reuse remains provisional and needs actual joint-change episodes plus complete observer, nonlinear simulation-error adaptation, replay and capacity controls. The next high-information step is the standard multi-step calibration control, with independent information-boundary and noise checks. A dataset-specific F16 improvement would not satisfy the general-world-model objective.',
 'final_research_direction.md':'No final direction accepted. Research Question remains provisional: when can learned dynamics be reused after unpaired observation changes and changing physical systems? Current Evidence demonstrates ordinary repairs, not a uniquely unresolved mechanism. Existing Methods include Wiener identification, nonlinear refits and latent dynamics residual adaptation. Proposed Idea: none validated. Falsification: complete ordinary calibration/refit/state-space/replay controls explain the alleged failure. Feasibility: current EMPS/F16 server data and CPU-qualified controls; C71 original-budget GPU experiment continues. Next Experiment: fixed-information simulation-error sensor calibration before a sequential joint-change test. Goal ACTIVE/UNACHIEVED.'}
for name,body in entries.items():
 with (ROOT/name).open('x',encoding='utf-8') as f:f.write('# '+name[:-3].replace('_',' ')+'\n\n'+body+'\n\nSee CALIBRATION_FINDING75_V1.md for full limits and provenance.\n')
def writecsv(path,rows):
 with path.open('x',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
aggregate=[];records=[];modelrows=[]
for r in result['rows']:
 aggregate.append({k:r[k] for k in ['dataset','condition','method','normalizer_calibration_x_std','all_records_finite','failed_records','rmse','nrmse','mae','arrays_path','arrays_sha256']})
 for rr in r['per_record']:records.append(dict(dataset=r['dataset'],condition=r['condition'],method=r['method'],**rr))
 modelrows.append(dict(dataset=r['dataset'],condition=r['condition'],method=r['method'],model=r['model']))
writecsv(OUT/'aggregate_metrics75_v1.csv',aggregate);writecsv(OUT/'record_metrics75_v1.csv',records);writecsv(OUT/'independent_replays75_v1.csv',verification['checks'])
dump(OUT/'ADAPTER_SELECTION75_V1.json',dict(raw_result_sha256=sha(OUT/'CALIBRATION_RESULTS75_V1.json'),all_candidates=result['adapter_selection']))
dump(OUT/'MODEL_METADATA75_V1.json',dict(raw_result_sha256=sha(OUT/'CALIBRATION_RESULTS75_V1.json'),models=modelrows))
compact={k:v for k,v in result.items() if k not in ['rows','adapter_selection','process_affinity']};compact['aggregate_rows']=aggregate;compact['neural_seed_summary']=neural;dump(OUT/'RESULT_SUMMARY75_V1.json',compact)
vq={k:v for k,v in verification.items() if k!='checks'};dump(OUT/'VERIFICATION_SUMMARY75_V1.json',vq)
closure=dict(utc=utc(),status='CLOSED_ORDINARY_CALIBRATION_AND_NONLINEAR_REFITS_NO_NEW_MECHANISM_ACCEPTED',raw_result_sha256=sha(OUT/'CALIBRATION_RESULTS75_V1.json'),verification_sha256=sha(OUT/'VERIFICATION75_V1.json'),terminal_sha256=sha(OUT/'run75_v1_exit.json'),protocol_sha256=result['protocol_sha256'],source_fits=2,calibration_ridge_fits=6,inverse_map_candidates=48,neural_fits=18,neural_updates=45000,aggregate_rows=54,record_rows=243,forecast_values=2011392,independent_replays=486,all_records_finite=True,official_test_values_parsed=False,new_method=False,accepted_final_directions=0,overall_goal='ACTIVE_UNACHIEVED',local_result_files_written=0)
dump(OUT/'CLOSURE75_V1.json',closure)
status='''# C75 current state: ordinary sensor calibration/refits tested

See cycle75/CALIBRATION_FINDING75_V1.md and cycle75/outputs/CLOSURE75_V1.json. Two real systems, nine original TRAIN records; source/calibration trajectories disjoint, source-only normalization. Identity/affine/monotone nonlinear observation effects are explicitly constructed, not measured sensor replacements or changed physical dynamics. Eighteen ordinary neural refits completed45,000 updates, plus two source/six calibration ridge fits and48 inverse-map candidates. All54 aggregate/243 record rows retained;2,011,392 forecast values and486 independent recurrences verified. No official test MAT output read.

EMPS nonlinear NRMSE: unadapted0.040125129, ordinary monotone calibration0.023289393, nonlinear refits0.010562858/0.014706616/0.020304213. F16 nonlinear: unadapted0.521722905, monotone calibration0.409961400, nonlinear refits0.349425181/0.327292385/0.342957464. These show ordinary repairs; the remaining F16 one-step-calibration residual has no new-mechanism attribution. Next control is standard simulation-error calibration with state/noise checks, before any new architecture or claimed sequential joint-change gap.

No final research direction accepted; general world-model research goal ACTIVE/UNACHIEVED. Independent C71 original-budget three-seed training and frozen assessment continue unchanged. Data/arrays/weights stay on server G; compact evidence/code/reports are selected for GitHub only, with no local result-file synchronization.
'''
with (PROJECT/'CURRENT_RESEARCH_STATUS_V75.md').open('x',encoding='utf-8') as f:f.write(status)
files=list(entries)+['CALIBRATION_FINDING75_V1.md','PROTOCOL75_V1.md','VERIFICATION_PROTOCOL75_V1.md']
files+=['scripts/'+p.name for p in sorted((ROOT/'scripts').glob('*.py'))]
files+=['outputs/'+n for n in ['DATA_SPLITS75_V1.json','ADAPTER_SELECTION75_V1.json','MODEL_METADATA75_V1.json','RESULT_SUMMARY75_V1.json','VERIFICATION_SUMMARY75_V1.json','aggregate_metrics75_v1.csv','record_metrics75_v1.csv','independent_replays75_v1.csv','CLOSURE75_V1.json','run75_v1_launch.json','run75_v1_exit.json','run75_v1_stdout.log']]
patterns=json.loads(os.environ['C75_PRIVATE_EXPORT_PATTERNS_JSON']);dest=OUT/'publication_v1';dest.mkdir(exist_ok=False);pubrows=[]
for rel in sorted(files):
 raw=(ROOT/rel).read_bytes()
 for needle in patterns:assert needle.encode() not in raw,(rel,'private export pattern')
 b=raw.decode('utf-8-sig').encode();target=dest/'files'/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(b);pubrows.append(dict(path=rel,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),git_blob_sha1=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest(),server_snapshot_file=str(target)))
manifest=dict(utc=utc(),status=closure['status'],files=pubrows,file_count=len(pubrows),bytes=sum(r['bytes'] for r in pubrows),raw_arrays_weights_included=False,official_test_values_parsed=False,raw_result_sha256=sha(OUT/'CALIBRATION_RESULTS75_V1.json'),raw_verification_sha256=sha(OUT/'VERIFICATION75_V1.json'),complete_metrics_and_candidates_split_into_compact_files=True,accepted_final_directions=0,overall_goal='ACTIVE_UNACHIEVED',local_result_files_written=0)
dump(dest/'manifest.json',manifest);print(json.dumps(dict(status=closure['status'],files=len(pubrows),bytes=manifest['bytes'],manifest_sha256=sha(dest/'manifest.json'))),flush=True)
