"""Close original-training-record controls with full scope and failure provenance."""
import sys,pathlib,json,hashlib,os
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common74_v1 import *
m=read(OUT/'TRAIN_DATA74_V2.json');r=read(OUT/'CANONICAL_ARX74_V2.json');q=read(OUT/'CANONICAL_VERIFICATION74_V1.json')
assert q['status']=='PASS_INDEPENDENT_COMPANION_ROLLOUT_AND_ALL_DEV_METRICS' and q['canonical_result_sha256']==sha(OUT/'CANONICAL_ARX74_V2.json')
assert len(m['records'])==9 and r['new_ridge_fits']==6 and q['forecast_value_rows_rechecked']==372480
assert read(OUT/'canonical74_v2_exit.json')['exit_code']==0 and read(OUT/'canonical74_v1_exit.json')['exit_code']==1
assert all(row['failed_records']==0 for row in r['rows']) and not m['official_test_numeric_values_parsed']
values={}
for row in r['rows']:
 assert sha(row['array_path'])==row['array_sha256'];values[(row['dataset'],row['method'])]=row['normalized_rmse']
table='| Control | EMPS DEV NRMSE | F16 DEV NRMSE |\n|---|---:|---:|\n'
for method in ['ridge_arx_lag4','ridge_arx_lag16','ridge_arx_lag64','persistence','train_mean']:table+='| '+method+' | '+format(values['EMPS',method],'.9f')+' | '+format(values['F16',method],'.9f')+' |\n'
report='''# C74: physical data and canonical dynamics controls

Two physical systems and nine original training records are now qualified. These are new ordinary-control executions, not a newly proposed world-model framework. No sensor-shift score or official-test score has been computed. No final research direction is accepted.

## Data and information

EMPS supplies one24841-sample original training record, using motor force vir and position qm. F16 supplies four73728-sample multisine records and four108477-sample swept-sine records, using Force and the official default Acceleration channel0. F16 Fs is400Hz. EMPS has strictly positive released clock increments near1ms; their tiny variation is retained without resampling. Keep raw source units; no unit inference from magnitude.

The two official public archives total150405224 bytes, downloaded with normal HTTPS verification. F16 was downloaded directly on the server. After the server's Google connection timed out, the identical official EMPS ZIP went through transient RAM and verified pinned SFTP; no data file was saved locally. Archive/source/member SHA256, sizes and ZIP CRCs are recorded. No publisher-provided checksum is claimed verified. Only training MAT members were extracted and parsed numerically; official test files remain in the archives. No photographs were extracted.

The bounded prior-ledger review checked136 baseline/result files. It found actual Silverbox scores in C29, so Silverbox was excluded. No EMPS/F16 score mention was found in that check; this is not a universal no-exposure certificate. The current independent corpus must retain that provenance limit.

EMPS was collected under feedback. Here the supplied recorded future motor-force path is an explicit condition of simulation. These scores do not identify the consequences of changing its controller or represent an operational forecast with unknown future actions. F16 uses the supplied measured shaker-force path. We do not merge these different experimental input mechanisms into a causal-effect claim.

## What was fit and scored

Each original TRAIN record is split at80% by time; the following256 samples separate its fit segment from DEV forecast origins. One pooled fixed train-only u/y mean and population SD is fit per physical system. Three float64 recursive ridge-ARX controls use lags4,16,64 and fixed regularizer0.001 times the mean non-intercept Gram diagonal. No future y enters a forecast. Each origin supplies128 history samples and128 future input values; all128 outputs are generated recursively without resets. There are582 complete origins,74496 forecast points per method, across the two systems. These are windows, not582 independent physical systems. The two systems have different physical-duration horizons; no cross-system score is averaged.

'''+table+'''
All records and all three lags are retained. DEV selects lag64 as the ordinary competence control in each system. All forecasts are finite. Full per-record raw/normalized metrics, counts, coefficient hashes, raw prediction hashes and all runtime details are in outputs/CANONICAL_ARX74_V2.json. Six ridge fits completed; no neural model was trained here. These are discovery DEV scores under a rolling-window task, not official full-record leaderboard scores or an independent test confirmation.

## Verification and engineering provenance

An independent companion-state matrix implementation reproduces54 first/last-window ARX rollouts to max absolute difference1.0658141036401503e-14. All372480 saved forecast-value rows across the five methods were rechecked against the canonical source arrays; saved target differences are exactlyzero and all reported DEV metrics reconstruct. Persistence/mean predictions and source/artifact hashes also match. The first preparation version failed its overly strict uniform-clock assertion before writing a canonical dataset or fitting a model. Its source and terminal log are preserved. The monotone-clock correction changes no split, forecast horizon, regularization or lag. The remote EMPS acquisition failure is also preserved.

## Research consequence

Both original observation interfaces already admit competent ordinary dynamics controls, especially EMPS. Their scores do not prove that observation/dynamics reuse is solved; no constructed interface change has yet been tested. They do prevent treating a weak uncalibrated predictor as the only comparison. F16's remaining error is not itself a research gap and may reflect linear-model limitations, noise or the selected rollout protocol.

The next experiment must freeze a declared measurement-shift curriculum and compare complete nonlinear calibration/observer, state-space, replay and no-reuse controls. Sensor operators will be explicitly constructed over real physical trajectories, not described as measured hardware replacements. Source/calibration raw segments must be disjoint if the setting excludes paired views; separate-device switches create separate episodes. Original official test outputs stay closed until all rules and method choices are fixed. The original C71 full-budget three-seed campaign and its separate frozen assessment remain unchanged. Overall goal ACTIVE/UNACHIEVED; accepted final directions0.
'''
with (ROOT/'CANONICAL_FINDING74_V1.md').open('x',encoding='utf-8') as f:f.write(report)
entries={
 'literature_review.md':'The source is the pinned official nonlinear_benchmarks loader and its primary EMPS/F16 data references. Current world-model overlap is recorded in cycle71/NOVELTY_BOUNDARY71_V4.md. No data-science gap follows merely from a new benchmark or constructed sensor transform.',
 'baseline_registry.md':'Actually executed: six deterministic ridge-ARX fits (lags4/16/64 per physical system), persistence and training-mean controls. Float64, fixed regularizer, pooled train-only scaling. Complete nonlinear observer/state-space/replay baselines remain required before a new mechanism claim. No neural fit or source-paper table reproduction claimed.',
 'experimental_results.md':'Nine original TRAIN records,602924 fit points,582 DEV origins with128 free-running steps. '+table+'\nAll record-level metrics and hashes retained. Six ridge fits; ten per-system method rows; zero official-test or sensor-shift scores.',
 'negative_evidence.md':'Ordinary ARX already yields EMPS DEV NRMSE0.020693531 and F16 DEV0.335650939 with lag64. A weak canonical baseline is not a sound premise for a new framework. These source-observation results neither test joint shifts nor show every dynamics model sufficient. Silverbox excluded after prior C29 scores were found. Preserve both acquisition and strict-clock preparation failures as engineering issues.',
 'hypothesis_space.md':'Open: can complete ordinary observation adapters and state-space/replay controls preserve learned dynamics under prospectively specified unpaired observation changes and system switches? Canonical competence is now measured. Unidentifiable sensor/state explanations require predictive uncertainty. F16 residual error alone cannot distinguish linear approximation, observation noise, training data or a reusable-mechanism failure.',
 'research_ideas.md':'Retain sequential observation/dynamics reuse as a provisional question. New data/control executions supply an independent starting point; they do not supply a distinct method. The next curriculum must separate physical-system identity, constructed sensor operator and replay information. No paired raw trace in disguise, future sensor value, new capacity or extra training may be given only to a candidate.',
 'final_research_direction.md':'No final direction accepted. EMPS/F16 training data and ordinary canonical controls are ready; independent verification passed. Next decisive work is a prospectively frozen measurement-shift/reuse experiment with complete mature controls, followed by independent official-test confirmation if a residual survives. No new mechanism is validated; full goal ACTIVE/UNACHIEVED.'}
for name,body in entries.items():
 with (ROOT/name).open('x',encoding='utf-8') as f:f.write('# '+name[:-3].replace('_',' ')+'\n\n'+body+'\n\nSee CANONICAL_FINDING74_V1.md and the machine-readable source, data, result and verification receipts.\n')
closure=dict(utc=utc(),status='CLOSED_CANONICAL_PHYSICAL_CONTROLS_SENSOR_REUSE_TEST_NOT_RUN',data_receipt_sha256=sha(OUT/'DATA_SOURCE74_V2.json'),training_manifest_sha256=sha(OUT/'TRAIN_DATA74_V2.json'),result_sha256=sha(OUT/'CANONICAL_ARX74_V2.json'),verification_sha256=sha(OUT/'CANONICAL_VERIFICATION74_V1.json'),records=9,new_ridge_fits=6,neural_fits=0,DEV_origins=582,official_test_numeric_values_parsed=False,sensor_shift_scores=0,accepted_final_directions=0,overall_goal='ACTIVE_UNACHIEVED',local_result_files_written=0)
dump(OUT/'CANONICAL_CLOSURE74_V1.json',closure)
status='''# C74 current state: new physical corpus and ordinary controls executed

See cycle74/CANONICAL_FINDING74_V1.md and cycle74/outputs/CANONICAL_CLOSURE74_V1.json. EMPS and F16 supplied nine original training records. Six ordinary ridge-ARX fits and persistence/mean controls generated582 DEV origins per method, each128 free-running steps. Independent companion-state and all-score reconstruction passed. Raw data and predictions stay on the configured server G storage. EMPS transfer used transient RAM after a remote network timeout; no local data file was saved.

The best declared lag64 control has DEV NRMSE0.020693531 on EMPS and0.335650939 on F16; all three lags and every record remain in the evidence. These are canonical observation controls, not a new method, sensor-shift evidence, causal-effect estimates or official leaderboard scores. Official test outputs have not been parsed numerically. A strict-clock preparation failure and failed direct EMPS acquisition are preserved.

The focused candidate remains sequential observation/dynamics reuse without paired views or future-task encoder exposure. A separate, prospectively frozen constructed-measurement curriculum and complete nonlinear observer/state-space/replay controls are still needed. No final research direction accepted; overall goal ACTIVE/UNACHIEVED. The independent C71 original-budget three-seed campaign and frozen assessment continue unchanged. GitHub receives selected authored code, reports and compact evidence; no local result synchronization.
'''
with (PROJECT/'CURRENT_RESEARCH_STATUS_V74.md').open('x',encoding='utf-8') as f:f.write(status)
files=list(entries)+['CANONICAL_FINDING74_V1.md','DATA_PROTOCOL74_V1.md','DATA_SCALING_CLARIFICATION74_V1.md','CLOCK_CLARIFICATION74_V1.md']
files+=['scripts/'+name for name in ['common74_v1.py','acquire74_v1.py','capture_acquisition74_v1.py','inspect_training74_v1.py','prepare74_v1.py','prepare74_v2.py','audit_clock74_v1.py','canonical_arx74_v1.py','canonical_arx74_v2.py','capture_canonical74_v1.py','capture_canonical74_v2.py','verify_canonical74_v1.py','export_canonical74_v1.py']]
files+=['outputs/'+name for name in ['ACQUISITION74_V1.json','DATA_SOURCE74_V2.json','DATA_EMPS74_V2.json','DATA_F1674_V1.json','FAILURE_EMPS74_V1.json','acquisition74_v1_launch.json','acquisition74_v1_exit.json','acquisition74_v1_stdout.log','canonical74_v1_prepare74_v1_launch.json','canonical74_v1_exit.json','canonical74_v1_stdout.log','TRAIN_CLOCK74_V1.json','TRAIN_DATA74_V2.json','CANONICAL_ARX74_V2.json','CANONICAL_VERIFICATION74_V1.json','CANONICAL_CLOSURE74_V1.json','canonical74_v2_prepare74_v2_launch.json','canonical74_v2_canonical_arx74_v2_launch.json','canonical74_v2_exit.json','canonical74_v2_stdout.log']]
patterns=json.loads(os.environ['C74_PRIVATE_EXPORT_PATTERNS_JSON']);staged=[]
for rel in sorted(files):
 raw=(ROOT/rel).read_bytes()
 for needle in patterns:assert needle.encode() not in raw,(rel,'Private export pattern')
 staged.append((rel,raw.decode('utf-8-sig').encode()))
dest=OUT/'publication_v1';dest.mkdir(exist_ok=False);rows=[]
for rel,b in staged:
 target=dest/'files'/rel;target.parent.mkdir(exist_ok=True,parents=True);target.write_bytes(b)
 rows.append(dict(path=rel,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),git_blob_sha1=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest(),server_snapshot_file=str(target)))
manifest=dict(utc=utc(),status=closure['status'],files=rows,file_count=len(rows),bytes=sum(r['bytes'] for r in rows),raw_data_weights_or_predictions_included=False,official_test_numeric_values_parsed=False,accepted_final_directions=0,overall_goal='ACTIVE_UNACHIEVED',local_result_files_written=0)
dump(dest/'manifest.json',manifest)
print(json.dumps(dict(status=closure['status'],files=len(rows),bytes=manifest['bytes'],manifest_sha256=sha(dest/'manifest.json'))),flush=True)
