"""Close and export only the terminal rollout qualification; retain live fits."""
import sys,pathlib,json,hashlib,datetime,os,math,base64
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common71_v1 import *

exit_record=read(OUT/'open_loop71_v1_exit.json')
assert exit_record['exit_code']==0
assert sha(OUT/'open_loop71_v1_stdout.log')==exit_record['log_sha256']
result=read(OUT/'OPEN_LOOP_QUALIFICATION71_V1.json')
assert result['status']=='PASS_COMMON_HISTORY_FREE_ROLLOUT_INTERFACE'
assert result['train_rows']==[0,1,2,3]
assert result['val_forecast_rows']==result['eval_forecast_rows']==0
assert result['module_updates']==result['proposal_updates']==0
assert not result['performance_test'] and not result['new_method']
assert result['protocol_sha256']==sha(ROOT/'OPEN_LOOP_PROTOCOL71_V1.md')
assert sha(result['array_path'])==result['array_sha256']
assert sha(result['checkpoint']['path'])==result['checkpoint']['sha256']
assert [r['horizon'] for r in result['horizons']]==[1,10,50]
assert all(r['finite'] and r['padding_max_abs']==0 for r in result['horizons'])
assert all(r['relative_scaled']<=2e-4 for r in result['numpy_reference'])
procs=json.loads(os.environ['C71_LIVE_PROCESSES_JSON'])
procs=procs if isinstance(procs,list) else [procs]
workers=[p for p in procs if 'train71_v3.py' in p['CommandLine']]
assert len(workers)==1,'Confirm one formal worker; do not restart it'
progress=[read(p) for p in sorted(OUT.glob('formal_seed*_v3_progress.json'))]
closure=dict(utc=utc(),status='CLOSED_INTERFACE_QUALIFICATION_BASELINE_FITS_STILL_RUNNING',qualification=result,qualification_result_sha256=sha(OUT/'OPEN_LOOP_QUALIFICATION71_V1.json'),capture_exit=exit_record,formal_processes=procs,formal_progress=progress,performance_claim=False,reserved_eval_scored=False,accepted_final_directions=0,overall_goal='ACTIVE_UNACHIEVED',local_result_data_or_model_files_written=0)
dump(OUT/'OPEN_LOOP_CLOSURE71_V1.json',closure)
report='''# Common-history future simulator: execution qualification

This closes a numeric/interface check, not model-performance evaluation. The three formal source-based subset fits remain in progress. No final research direction has been accepted.

The native model strictly loaded all 3,202,310 parameters from the first fit's immutable epoch-50 development checkpoint. Using only TRAIN rows 0-3, unchanged native proposals inferred graph assignments from 49 observed steps. The frozen composer then generated 1, 10 and 50 future steps from the last observed state. Neither future targets nor true graph labels supplied the inference loss.

All scored outputs were finite. Replacing future padding 0 with 111 changed no scored prediction. Independently evaluated NumPy float64 neural modules agreed with the native float32 50-step rollout; exact errors and hashes are in OPEN_LOOP_CLOSURE71_V1.json. Checkpoint and model state hashes remained unchanged, and both optimizer-update counts were zero.

This establishes that the tested interface can produce a genuine common-history free rollout. It says nothing about predictive accuracy or generalization. The checkpoint is an early development checkpoint, not the final selected model. No VAL forecasting errors or reserved EVAL scores were computed. The original native reset-block VAL metric remains the declared formal checkpoint-selection criterion.

Native source bodies, checkpoint/result/log hashes and raw-array location are recorded. Raw histories, graphs and predictions remain on server G storage. This GitHub export includes source, closed metadata/logs and this report; it excludes raw arrays, weights and live training logs. NOVELTY_BOUNDARY71_V2.md records additional overlap with existing module-reuse methods. The next method question still requires an independently specified residual and strong ordinary controls.
'''
with (OUT/'OPEN_LOOP_VERIFICATION71_V1.md').open('x',encoding='utf-8') as f:f.write(report)
status='''# C71 update: free-rollout interface qualified; formal fits running

See cycle71/outputs/OPEN_LOOP_VERIFICATION71_V1.md and OPEN_LOOP_CLOSURE71_V1.json for the closed numeric/interface qualification and the bounded live-fit snapshot. No model accuracy was evaluated in that qualification. All three formal fits remain under the original frozen protocol; reserved EVAL remains closed.

New primary-source overlap checks in cycle71/NOVELTY_BOUNDARY71_V2.md rule out generic lightweight world-model stitching and frozen-module latent interfaces as new claims. No final research direction accepted; overall goal ACTIVE/UNACHIEVED. Data, weights and raw arrays stay on authorized server G storage. Only source, reports and compact evidence are uploaded to GitHub.
'''
with (PROJECT/'CURRENT_RESEARCH_STATUS_V71_OPEN_LOOP.md').open('x',encoding='utf-8') as f:f.write(status)
dest=OUT/'publication_open_loop_v1';dest.mkdir(exist_ok=False)
selected=['OPEN_LOOP_PROTOCOL71_V1.md','NOVELTY_BOUNDARY71_V2.md','scripts/open_loop71_v1.py','scripts/capture_open_loop71_v1.py','scripts/export_open_loop71_v1.py','start_open_loop71_v1.ps1','outputs/OPEN_LOOP_QUALIFICATION71_V1.json','outputs/OPEN_LOOP_CLOSURE71_V1.json','outputs/OPEN_LOOP_VERIFICATION71_V1.md','outputs/open_loop71_v1_launch.json','outputs/open_loop71_v1_exit.json','outputs/open_loop71_v1_stdout.log']
patterns=json.loads(os.environ['C71_PRIVATE_EXPORT_PATTERNS_JSON']);rows=[]
for rel in selected:
 p=ROOT/rel;raw=p.read_bytes();original_rel=rel
 for needle in patterns:assert needle.encode() not in raw,(rel,'private export pattern')
 try:body=raw.decode('utf-8-sig');encoding='UTF-8'
 except UnicodeDecodeError:body=base64.b64encode(raw).decode('ascii');rel+='.base64.txt';encoding='Base64 exact original log bytes'
 b=body.encode();target=dest/'files'/rel;target.parent.mkdir(exist_ok=True,parents=True);target.write_bytes(b)
 rows.append(dict(path=rel,original_path=original_rel,original_sha256=hashlib.sha256(raw).hexdigest(),encoding=encoding,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),git_blob_sha1=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest(),parts=math.ceil(len(body)/3500),server_snapshot_file=str(target)))
manifest=dict(utc=utc(),status=closure['status'],files=rows,file_count=len(rows),bytes=sum(r['bytes'] for r in rows),closure_sha256=sha(OUT/'OPEN_LOOP_CLOSURE71_V1.json'),raw_arrays_weights_and_live_logs_included=False,reserved_eval_scored=False,accepted_final_directions=0,overall_goal='ACTIVE_UNACHIEVED',local_result_data_or_model_files_written=0)
dump(dest/'manifest.json',manifest)
print(json.dumps(dict(status=closure['status'],files=len(rows),bytes=manifest['bytes'],manifest_sha256=sha(dest/'manifest.json'),live_worker=workers[0]['ProcessId'],progress=progress)),flush=True)
