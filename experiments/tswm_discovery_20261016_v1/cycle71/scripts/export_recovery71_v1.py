"""Snapshot truthful driver-incident recovery, prior-art screen and live dependencies."""
import sys,pathlib,ast,math,base64,os
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common71_v1 import *
incident=read(OUT/'TRAINING_INCIDENT71_V1.json');q=read(OUT/'RESUME_QUALIFICATION71_V2.json');assert q['status']=='PASS_CHECKPOINT_CONTINUATION_EQUIVALENCE' and q['comparison']['max_abs']==0 and q['comparison']['non_tensor_mismatches']==0
assert read(OUT/'recovery71_v2_qualify_exit.json')['exit_code']==0
assert sha(incident['last_checkpoint']['path'])==incident['last_checkpoint']['sha256']
for row in [q['full_checkpoint'],q['resumed_checkpoint']]:assert sha(row['path'])==row['sha256']
assert not (OUT/'free_simulation71_v1_launch.json').exists() and not (OUT/'free_simulation71_v2_launch.json').exists()
source=read(OUT/'SOURCE_AUDIT71_LEJEPA_V2.json')
for row in source['files']:assert sha(row['server_path'])==row['sha256']
base=read(OUT/'PROGRESS_SNAPSHOT71_V1.json')['reports'];tree=ast.parse((ROOT/'scripts'/'export_novelty71_v3.py').read_bytes());update_node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='updates' for t in n.targets));updates=ast.literal_eval(update_node.value)
itree=ast.parse((ROOT/'scripts'/'audit_training_incident71_v1.py').read_bytes());incident_suffix=next(ast.literal_eval(n.value) for n in ast.walk(itree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='suffix' for t in n.targets))
recovery_suffix=b'\n\n## Verified checkpoint continuation, C71\n\nRESUME_QUALIFICATION71_V2.json reports zero differences across174 tensors and all compared optimizer/RNG/shuffle/structure/metric values for uninterrupted versus resumed qualification runs. The first formal fit resumed from epoch650 under RECOVERY_PROTOCOL71_V1.md, preserving the original loop body and checkpoint grid. The recovered campaign and its new assessment queue have new process/receipt identities. See RECOVERY_SNAPSHOT71_V1.json for the bounded live state. Root cause of the NVIDIA-driver exception remains unestablished; no final model accuracy or research-direction claim is made.\n'
for name,body in updates.items():
 p=ROOT/name;raw=p.read_bytes();old=raw[:-len(recovery_suffix)] if raw.endswith(recovery_suffix) else raw;novelty_suffix=('\n\n## Primary-source refinement, C71 v3\n\n'+body+'\n').encode();tail=novelty_suffix+incident_suffix;assert old.endswith(tail) and hashlib.sha256(old[:-len(tail)]).hexdigest()==base[name],(name,'Unanticipated ledger mutation')
 if not raw.endswith(recovery_suffix):
  with p.open('ab') as f:f.write(recovery_suffix)
procs=json.loads(os.environ['C71_LIVE_PROCESSES_JSON']);procs=procs if isinstance(procs,list) else [procs]
workers=[p for p in procs if '\\scripts\\resume_native71_v2.py formal' in p['CommandLine']];queues=[p for p in procs if '\\scripts\\queue_free_simulation71_v2.py' in p['CommandLine']];assert len(workers)==len(queues)==1
assert workers[0]['ProcessId']==read(OUT/'recovery71_v2_formal_launch.json')['pid'] and queues[0]['ProcessId']==read(OUT/'queue_free_simulation71_v2_launch.json')['pid']
progress=read(OUT/'formal_seed2026711_resumed_v2_progress.json');assert progress['epoch']>=653 and progress['module_updates']==(progress['epoch']+1)*16
obj=dict(utc=utc(),status='RECOVERED_FORMAL_FIT_AND_ASSESSMENT_QUEUE_RUNNING_NOT_COMPLETE',live_processes=procs,progress=progress,incident_sha256=sha(OUT/'TRAINING_INCIDENT71_V1.json'),resume_qualification_sha256=sha(OUT/'RESUME_QUALIFICATION71_V2.json'),recovery_protocol_sha256=sha(ROOT/'RECOVERY_PROTOCOL71_V1.md'),source_audit_sha256=sha(OUT/'SOURCE_AUDIT71_LEJEPA_V2.json'),ledger_hashes={name:sha(ROOT/name) for name in updates},neural_free_val_assessment_started=False,driver_root_cause_verified=False,reserved_eval_scored=False,accepted_final_directions=0,overall_goal='ACTIVE_UNACHIEVED',local_result_data_or_model_files_written=0)
dump(OUT/'RECOVERY_SNAPSHOT71_V1.json',obj)
log=OUT/'formal_seed2026711_v3_stdout.log';raw=log.read_bytes();start=raw.rfind(b'Windows fatal exception:');assert start>=0;excerpt=raw[start:];encoded=base64.b64encode(excerpt).decode('ascii')
with (OUT/'DRIVER_CRASH_EXCERPT71_V1.base64.txt').open('x',encoding='ascii') as f:f.write(encoded)
dump(OUT/'DRIVER_CRASH_EXCERPT71_V1.json',dict(full_log_sha256=sha(log),full_log_bytes=len(raw),excerpt_byte_offset=start,excerpt_bytes=len(excerpt),excerpt_sha256=hashlib.sha256(excerpt).hexdigest(),encoding='Base64 exact terminal log excerpt',full_log_remains_on_server=True))
status='''# C71: first fit recovered after a recorded NVIDIA-driver exception

The original first fit terminated with Windows access violation0xc0000005; WER names nvcuda64.dll. Original campaign/queue stopped and no free-VAL assessment launched. The last epoch650 checkpoint passed a174-tensor finite-value check. The root cause remains unestablished.

The continuation qualification matches the uninterrupted reference exactly across174 tensors and all compared optimizer/RNG/shuffle/structure/metric values. The original epoch-loop body is unchanged. The recovered first fit has passed the earlier failure point; see cycle71/outputs/RECOVERY_SNAPSHOT71_V1.json for the verified process/time/update snapshot. Two fresh original seeds follow it, then the frozen assessment. CUDA_LAUNCH_BLOCKING=1 is diagnostic execution, not a proven driver fix. This is not completed training.

See cycle71/RECOVERY_PROTOCOL71_V1.md for lineage/update accounting, cycle71/NOVELTY_BOUNDARY71_V3.md for primary-source novelty refinements, and the seven living ledgers for corrected status. No final research direction accepted; full goal ACTIVE/UNACHIEVED. EVAL stays closed. Raw data/arrays/weights stay on serverG; GitHub receives authored code, reports and compact closed evidence.
'''
with (PROJECT/'CURRENT_RESEARCH_STATUS_V71_RECOVERY.md').open('x',encoding='utf-8') as f:f.write(status)
closed=['TRAINING_INCIDENT71_V1.json','SOURCE_AUDIT71_LEJEPA_V2.json','RESUME_QUALIFICATION71_V2.json','RECOVERY_SNAPSHOT71_V1.json','DRIVER_CRASH_EXCERPT71_V1.base64.txt','DRIVER_CRASH_EXCERPT71_V1.json','formal_seed2026711_v3_exit.json','campaign71_v3_exit.json','queue_free_simulation71_v1_exit.json','recovery71_v1_qualify_launch.json','recovery71_v1_qualify_exit.json','recovery71_v2_qualify_launch.json','recovery71_v2_qualify_exit.json','recovery71_v1_qualify_stdout.log','recovery71_v2_qualify_stdout.log','resume_full_control71_v1_result.json','campaign71_v4_launch.json','recovery71_v2_formal_launch.json','queue_free_simulation71_v2_launch.json','FREE_SIMULATION_QUEUE71_V2.json','ASSESSMENT_RUNTIME71_V2.json']
files=[p for p in ROOT.iterdir() if p.is_file() and p.suffix in ('.md','.json','.ps1')]+list((ROOT/'scripts').glob('*.py'))+[OUT/name for name in closed]
dest=OUT/'publication_recovery_v1';dest.mkdir(exist_ok=False);rows=[];patterns=json.loads(os.environ['C71_PRIVATE_EXPORT_PATTERNS_JSON'])
for p in sorted(set(files)):
 raw=p.read_bytes();rel=p.relative_to(ROOT).as_posix();original=rel
 for needle in patterns:assert needle.encode() not in raw,(rel,'private export pattern')
 try:body=raw.decode('utf-8-sig');encoding='UTF-8'
 except UnicodeDecodeError:body=base64.b64encode(raw).decode('ascii');rel+='.base64.txt';encoding='Base64 exact original log bytes'
 b=body.encode();target=dest/'files'/rel;target.parent.mkdir(exist_ok=True,parents=True);target.write_bytes(b);rows.append(dict(path=rel,original_path=original,original_sha256=hashlib.sha256(raw).hexdigest(),encoding=encoding,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),git_blob_sha1=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest(),parts=math.ceil(len(body)/3500),server_snapshot_file=str(target)))
manifest=dict(utc=utc(),status=obj['status'],files=rows,file_count=len(rows),bytes=sum(r['bytes'] for r in rows),closure_sha256=sha(OUT/'RECOVERY_SNAPSHOT71_V1.json'),raw_arrays_weights_and_live_logs_included=False,reserved_eval_scored=False,accepted_final_directions=0,overall_goal='ACTIVE_UNACHIEVED',local_result_data_or_model_files_written=0)
dump(dest/'manifest.json',manifest);print(json.dumps(dict(status=obj['status'],files=len(rows),bytes=manifest['bytes'],manifest_sha256=sha(dest/'manifest.json'),progress=progress)),flush=True)
