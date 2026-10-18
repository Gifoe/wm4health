"""Publish the frozen assessment and closed TRAIN calibration before VAL execution."""
import sys,pathlib,json,math,base64,os
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common71_v1 import *

assert not (OUT/'free_simulation71_v1_launch.json').exists(),'Must freeze this publication before VAL assessment'
assert read(OUT/'history_controls71_v1_exit.json')['exit_code']==0
cal=read(OUT/'HISTORY_CONTROL_TRAIN_CALIBRATION71_V1.json');num=read(OUT/'HISTORY_CONTROL_NUMERICAL_AUDIT71_V1.json');graphs=read(OUT/'GRAPH_COVERAGE71_V1.json');queue=read(OUT/'FREE_SIMULATION_QUEUE71_V1.json')
assert cal['val_rows_accessed']==cal['eval_rows_accessed']==graphs['eval_rows_accessed']==0
assert sha(ROOT/'FREE_SIMULATION_PROTOCOL71_V1.md')==cal['protocol_sha256']==queue['protocol_sha256']
assert sha(ROOT/'scripts'/'evaluate_free_simulation71_v1.py')==queue['evaluator_sha256']
assert sha(OUT/'HISTORY_CONTROL_TRAIN_CALIBRATION71_V1.json')==num['calibration_sha256']==queue['calibration_sha256']
assert all(r['forecast_scaled_error']<1e-7 and r['relative_mse_difference']<1e-4 for r in num['checks'])
assert sha(cal['raw_artifact_path'])==cal['raw_artifact_sha256'] and sha(graphs['raw_artifact']['path'])==graphs['raw_artifact']['sha256']
assert sha(OUT/'history_controls71_v1_stdout.log')==read(OUT/'history_controls71_v1_exit.json')['log_sha256']
procs=json.loads(os.environ['C71_LIVE_PROCESSES_JSON']);procs=procs if isinstance(procs,list) else [procs]
queue_workers=[p for p in procs if '\\scripts\\queue_free_simulation71_v1.py' in p['CommandLine']]
train_workers=[p for p in procs if '\\scripts\\train71_v3.py' in p['CommandLine']]
assert len(queue_workers)==len(train_workers)==1
assert queue_workers[0]['ProcessId']==read(OUT/'queue_free_simulation71_v1_launch.json')['pid']
progress=[read(p) for p in sorted(OUT.glob('formal_seed*_v3_progress.json'))]
obj=dict(utc=utc(),status='TRAIN_CALIBRATION_CLOSED_VAL_ASSESSMENT_QUEUED_NOT_EXECUTED',live_processes=procs,progress=progress,queue=queue,calibration_sha256=sha(OUT/'HISTORY_CONTROL_TRAIN_CALIBRATION71_V1.json'),numerical_audit_sha256=sha(OUT/'HISTORY_CONTROL_NUMERICAL_AUDIT71_V1.json'),graph_audit_sha256=sha(OUT/'GRAPH_COVERAGE71_V1.json'),neural_free_val_assessment_started=False,reserved_eval_scored=False,accepted_final_directions=0,overall_goal='ACTIVE_UNACHIEVED',local_result_data_or_model_files_written=0)
dump(OUT/'ASSESSMENT_PLAN_CLOSURE71_V1.json',obj)
report='''# Free-simulation development assessment: protocol frozen, execution queued

The complete native three-seed training campaign is still running. A live server process holds a Windows synchronization handle to the existing campaign process and will launch the frozen assessment only after the process terminates successfully and all three terminal fit receipts and selected-checkpoint hashes pass verification. The queue does not restart training. The actual process identities and bounded training snapshot are in ASSESSMENT_PLAN_CLOSURE71_V1.json. No common-history VAL forecasting score has been produced by this assessment yet; reserved EVAL stays closed.

FREE_SIMULATION_PROTOCOL71_V1.md fixes six ordinary forecasting arms plus a separately marked privileged graph diagnostic. All start from the same 49 observed steps and simulate 50 future steps. The native model is evaluated with direct encoder assignments and 100/1000 native search steps, separating inference effort from fixed learned dynamics. Ordinary affine residual VAR(1)/(2) controls and persistence test whether a complex new mechanism is even needed. No future reset is allowed.

The ordinary regression controls completed all 2048 TRAIN trajectories and the six declared regularization values at each lag. Both selected alpha=1e-8; all grid evaluations were finite. These TRAIN suffix errors are calibration results, not independent generalization scores, and cannot be compared directly to native reset-block VAL scores. Exact grid results, normalization, qualification and source hashes are in HISTORY_CONTROL_TRAIN_CALIBRATION71_V1.json. No VAL/EVAL observations selected these controls.

High condition numbers triggered a targeted numerical check. On the worst-conditioned TRAIN history at each selected lag, normal-equation forecasts and independent augmented least squares differ by less than 3e-10 over 25 steps; relative MSE differences are below 9e-8. The measured errors are in HISTORY_CONTROL_NUMERICAL_AUDIT71_V1.json. No implementation change was required.

GRAPH_COVERAGE71_V1.json confirms 881 distinct TRAIN graphs and 230 distinct VAL graphs. Of 256 VAL trajectories, 214 (83.59375%) share their graph with TRAIN; 193 of 230 distinct VAL graphs have appeared in TRAIN. All inspected graphs are symmetric. The other 42 trajectories have a nearest TRAIN graph differing at two directed entries, or one undirected edge. Consequently, this development split measures trajectories with largely familiar structure. It cannot support an unseen-mechanism-composition claim. This is a protocol characterization, not an implementation defect or new scientific gap.

Next evidence: complete the actual models, execute the already frozen assessment, and attribute any deficit using inference-compute and privileged-graph controls before designing a new composition experiment. No final research direction accepted. Raw arrays, weights and calibration artifacts stay in authorized server G storage; GitHub receives code and compact closed evidence. The assessment's numerical performance remains unmeasured at this publication.
'''
with (OUT/'ASSESSMENT_PLAN71_V1.md').open('x',encoding='utf-8') as f:f.write(report)
status='''# C71: classical controls fixed; full free-simulation assessment queued

See cycle71/outputs/ASSESSMENT_PLAN71_V1.md and ASSESSMENT_PLAN_CLOSURE71_V1.json. Three native subset fits are still running. The frozen common-history development assessment is queued behind their successful termination and hash verification. It has not produced neural VAL forecasting scores. EVAL remains sealed.

TRAIN-only ordinary controls passed independent numerical checks. A graph audit shows 214/256 VAL trajectories share their structure with TRAIN, preventing an unseen-composition interpretation. No final research direction accepted; overall goal ACTIVE/UNACHIEVED. Raw arrays/data/weights remain on server G storage; only code, reports and compact evidence go to GitHub.
'''
with (PROJECT/'CURRENT_RESEARCH_STATUS_V71_ASSESSMENT.md').open('x',encoding='utf-8') as f:f.write(status)
selected=['FREE_SIMULATION_PROTOCOL71_V1.md','scripts/history_controls71_v1.py','scripts/prepare_history_controls71_v1.py','scripts/capture_history_controls71_v1.py','scripts/audit_history_controls71_v1.py','scripts/audit_graph_coverage71_v1.py','scripts/evaluate_free_simulation71_v1.py','scripts/queue_free_simulation71_v1.py','scripts/capture_assessment71_v1.py','scripts/export_assessment_plan71_v1.py','start_history_controls71_v1.ps1','start_queue_assessment71_v1.ps1','outputs/HISTORY_CONTROL_TRAIN_CALIBRATION71_V1.json','outputs/HISTORY_CONTROL_NUMERICAL_AUDIT71_V1.json','outputs/GRAPH_COVERAGE71_V1.json','outputs/FREE_SIMULATION_QUEUE71_V1.json','outputs/ASSESSMENT_PLAN_CLOSURE71_V1.json','outputs/ASSESSMENT_PLAN71_V1.md','outputs/history_controls71_v1_launch.json','outputs/history_controls71_v1_exit.json','outputs/history_controls71_v1_stdout.log','outputs/queue_free_simulation71_v1_launch.json']
dest=OUT/'publication_assessment_plan_v1';dest.mkdir(exist_ok=False);patterns=json.loads(os.environ['C71_PRIVATE_EXPORT_PATTERNS_JSON']);rows=[]
for rel in selected:
 p=ROOT/rel;raw=p.read_bytes();original_rel=rel
 for needle in patterns:assert needle.encode() not in raw,(rel,'private export pattern')
 try:body=raw.decode('utf-8-sig');encoding='UTF-8'
 except UnicodeDecodeError:body=base64.b64encode(raw).decode('ascii');rel+='.base64.txt';encoding='Base64 exact original log bytes'
 b=body.encode();target=dest/'files'/rel;target.parent.mkdir(exist_ok=True,parents=True);target.write_bytes(b)
 rows.append(dict(path=rel,original_path=original_rel,original_sha256=hashlib.sha256(raw).hexdigest(),encoding=encoding,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),git_blob_sha1=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest(),parts=math.ceil(len(body)/3500),server_snapshot_file=str(target)))
manifest=dict(utc=utc(),status=obj['status'],files=rows,file_count=len(rows),bytes=sum(r['bytes'] for r in rows),closure_sha256=sha(OUT/'ASSESSMENT_PLAN_CLOSURE71_V1.json'),raw_arrays_weights_and_live_logs_included=False,reserved_eval_scored=False,accepted_final_directions=0,overall_goal='ACTIVE_UNACHIEVED',local_result_data_or_model_files_written=0)
dump(dest/'manifest.json',manifest);print(json.dumps(dict(status=obj['status'],files=len(rows),bytes=manifest['bytes'],manifest_sha256=sha(dest/'manifest.json'),live_queue_pid=queue_workers[0]['ProcessId'],live_train_pid=train_workers[0]['ProcessId'],progress=progress)),flush=True)
