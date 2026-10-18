"""Snapshot immutable development evidence while the formal worker remains live."""
import sys,pathlib,json,hashlib,datetime,os,math,base64
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common71_v1 import *
assert read(OUT/'qualification71_v3_exit.json')['exit_code']==0 and read(OUT/'GRADIENT_AUDIT71_V1.json')['status']=='PASS'
procs=json.loads(os.environ['C71_LIVE_PROCESSES_JSON']);procs=procs if isinstance(procs,list) else [procs];workers=[p for p in procs if 'train71_v3.py' in p['CommandLine']];assert len(workers)==1,'Snapshot requires a currently verified formal worker'
catalog=read(OUT/'HISTORICAL_COVERAGE71_V1.json')
for row in catalog['cycles']:
 for s in row['sources'].values():assert sha(PROJECT/s['path'])==s['sha256']
public_index=dict(utc=utc(),status='PRIOR_DECISION_SOURCE_INDEX',full_server_index_sha256=sha(OUT/'HISTORICAL_COVERAGE71_V1.json'),cycle_count=catalog['cycle_count'],entries=[dict(cycle=r['cycle'],sources=r['sources'],question=r['question'],decision=r['decision'][:600]) for r in catalog['cycles']],new_experiments=0)
dump(OUT/'PRIOR_DECISION_INDEX71_V1.json',public_index)
progress=[]
for p in sorted(OUT.glob('formal_seed*_v3_progress.json')):
 r=read(p);progress.append(r)
snapshot=dict(utc=utc(),status='BASELINE_DEVELOPMENT_RUNNING_NOT_CLOSED',processes=procs,progress=progress,qualification_sha256=sha(OUT/'QUALIFICATION71_V3.json'),gradient_audit_sha256=sha(OUT/'GRADIENT_AUDIT71_V1.json'),acquisition_sha256=sha(OUT/'ACQUISITION71_V1.json'),protocol_sha256=sha(ROOT/'PROTOCOL71_V1.md'),reports={p.name:sha(p) for p in ROOT.glob('*.md')},actual_qualification_module_parameters=467204,actual_qualification_proposal_parameters=2735106,reserved_eval_scored=False,new_method=False,accepted_final_directions=0,overall_goal='ACTIVE_UNACHIEVED',raw_result_data_or_model_files_synchronized_locally=0,authored_local_report_drafts_removed_after_verified_upload=4)
dump(OUT/'PROGRESS_SNAPSHOT71_V1.json',snapshot)
with (PROJECT/'CURRENT_RESEARCH_STATUS_V71.md').open('x',encoding='utf-8') as f:f.write('''# C71: native modular-relational baseline development running

The native3.202M-parameter model passed an execution qualification and an actual label-invariant gradient/step check. Three fresh2000-epoch source-based subset fits are queued sequentially on the5090; see cycle71/outputs/PROGRESS_SNAPSHOT71_V1.json for the verified process/time and update counts at publication. These are not completed fits. DevelopmentVAL is a native reset-block score, not common-history full simulation. ReservedEVAL remains unscored.

Candidate observation/dynamics reuse and continual representation compatibility remain provisional. Earlier work and new source audits rule out generic modularity/partial-observation claims. No final new framework accepted; overall goal ACTIVE/UNACHIEVED. Raw data, weights and arrays remain in authorized serverGstorage. GitHub receives authored source, reports and this bounded development snapshot; no raw result/model/data files were synchronized locally.
''')
patterns=json.loads(os.environ['C71_PRIVATE_EXPORT_PATTERNS_JSON']);dest=OUT/'publication_progress_v1';dest.mkdir(exist_ok=False);rows=[]
closed=['ACQUISITION71_V1.json','TRANSFER_AUDIT71_V1.json','QUALIFICATION71_V3.json','GRADIENT_AUDIT71_V1.json','PRIOR_DECISION_INDEX71_V1.json','PROGRESS_SNAPSHOT71_V1.json']
files=[p for p in ROOT.iterdir() if p.is_file() and p.suffix in ['.md','.json','.ps1']]+list((ROOT/'scripts').glob('*.py'))+[OUT/n for n in closed]+list(OUT.glob('qualification71_v*_launch.json'))+list(OUT.glob('qualification71_v*_exit.json'))+list(OUT.glob('qualification71_v*_result.json'))+list(OUT.glob('qualification71_v*_stdout.log'))+[OUT/'campaign71_v3_launch.json']+list(OUT.glob('formal_seed*_v3_launch.json'))
for p in sorted(set(files)):
 raw=p.read_bytes();rel=p.relative_to(ROOT).as_posix();original_rel=rel
 for needle in patterns:assert needle.encode() not in raw,(rel,'private export pattern')
 try:text=raw.decode('utf-8-sig');encoding='UTF-8'
 except UnicodeDecodeError:text=base64.b64encode(raw).decode('ascii');rel+='.base64.txt';encoding='Base64 exact original log bytes'
 b=text.encode();target=dest/'files'/rel;target.parent.mkdir(exist_ok=True,parents=True);target.write_bytes(b);rows.append(dict(path=rel,original_path=original_rel,original_sha256=hashlib.sha256(raw).hexdigest(),encoding=encoding,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),git_blob_sha1=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest(),parts=math.ceil(len(text)/3500),server_snapshot_file=str(target)))
manifest=dict(utc=utc(),status='C71_SOURCE_QUALIFICATION_AND_RUNNING_SNAPSHOT_NOT_FINAL',files=rows,file_count=len(rows),bytes=sum(r['bytes'] for r in rows),progress_snapshot_sha256=sha(OUT/'PROGRESS_SNAPSHOT71_V1.json'),raw_arrays_weights_and_live_logs_included=False,reserved_eval_scored=False,accepted_final_directions=0,overall_goal='ACTIVE_UNACHIEVED',local_raw_result_data_or_model_files_written=0)
dump(dest/'manifest.json',manifest);print(json.dumps(dict(status=manifest['status'],files=len(rows),bytes=manifest['bytes'],manifest_sha256=sha(dest/'manifest.json'),live_worker=workers[0]['ProcessId'],progress=progress)),flush=True)
