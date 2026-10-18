"""Close the second terminal runtime fault without asserting a driver-only cause."""
import sys,pathlib,base64,subprocess,time
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common71_v1 import *
import numpy as np
import torch
torch.set_num_threads(1)
tag='formal_seed2026711_resumed_v2'
fit_exit=read(OUT/'recovery71_v2_formal_exit.json')
assert fit_exit['pid']==43972 and fit_exit['exit_code']==3221225477
log=OUT/'recovery71_v2_formal_stdout.log'
assert sha(log)==fit_exit['log_sha256']
campaign=read(OUT/'campaign71_v4_exit.json');queue=read(OUT/'queue_free_simulation71_v2_exit.json')
assert campaign['exit_code']==queue['exit_code']==1
for name in ['free_simulation71_v1_launch.json','free_simulation71_v2_launch.json']:
    assert not (OUT/name).exists()
manifest=read(OUT/(tag+'_checkpoints.json'));record=manifest['checkpoints'][-1]
assert record['epoch']==800 and record['module_updates']==record['proposal_updates']==12816
assert sha(record['path'])==record['sha256']
state=torch.load(record['path'],map_location='cpu',weights_only=False)
counts=dict(tensors=0,nonfinite=0)
def inspect(v):
    if isinstance(v,torch.Tensor):
        counts['tensors']+=1
        if v.is_floating_point() or v.is_complex():counts['nonfinite']+=int((~torch.isfinite(v)).sum())
    elif isinstance(v,dict):
        for x in v.values():inspect(x)
    elif isinstance(v,(list,tuple)):
        for x in v:inspect(x)
inspect(state);assert counts['nonfinite']==0
first=read(OUT/'TRAINING_INCIDENT71_V1.json')
values={};sources=[]
for source_tag,lo,hi in [('formal_seed2026711_v3',0,650),(tag,651,800)]:
    p=storage('artifact')/source_tag/'logs'/'seed2026711'/'scalars.jsonl'
    for line in p.open(encoding='utf-8'):
        row=json.loads(line)
        if row['tag']=='SA/temp' and lo<=row['step']<=hi:
            assert row['step'] not in values;values[row['step']]=row['value']
    sources.append(dict(tag=source_tag,epoch_start=lo,epoch_end=hi,sha256=sha(p)))
assert len(values)==801 and set(values)==set(range(801))
temp=np.exp(-9.8);trace=[]
for step in range(801):
    options=[temp*1.1,max(temp/1.1,np.exp(-100.))]
    matches=[j for j,v in enumerate(options) if float(np.log10(v))==values[step]]
    assert len(matches)==1,(step,matches)
    temp=options[matches[0]];trace.append(matches[0])
assert float(np.log10(temp))==state['metrics']['SA/temp']
event=json.loads(os.environ['C71_SECOND_EVENT_JSON']);assert event['pid']==43972
assert event['module']=='mtrand.cp312-win_amd64.pyd' and event['exception_code']=='c0000005'
raw=log.read_bytes();offset=raw.rfind(b'Windows fatal exception:');assert offset>=0
excerpt=raw[offset:]
with (OUT/'RUNTIME_CRASH_EXCERPT71_V2.base64.txt').open('x',encoding='ascii') as f:
    f.write(base64.b64encode(excerpt).decode('ascii'))
result=dict(utc=utc(),status='SECOND_NATIVE_ACCESS_VIOLATION_TERMINAL_CHECKPOINT_FINITE',
    first_incident_sha256=sha(OUT/'TRAINING_INCIDENT71_V1.json'),
    fit_exit=fit_exit,campaign_exit=campaign,queue_exit=queue,windows_application_error=event,
    first_fault_module=first['windows_error_reporting_fault_module'],
    second_fault_module=event['module'],exception_hex='0xc0000005',
    current_thread_source='hgnn_composer.py:350, np.random.choice inside native node-edge proposal',
    cuda_launch_blocking_was_enabled=True,diagnostic_synchronization_did_not_prevent_recurrence=True,
    root_cause='Unestablished: first fault WER named nvcuda64.dll; second named NumPy mtrand. Neither record establishes a software defect or hardware cause.',
    last_checkpoint=record,checkpoint_tensor_audit=counts,
    annealing_temperature_exact_replay=dict(steps=801,value=float(temp),float64_hex=float(temp).hex(),
        trace_sha256=hashlib.sha256(bytes(trace)).hexdigest(),source_logs=sources),
    terminal_excerpt=dict(full_log_bytes=len(raw),full_log_sha256=sha(log),byte_offset=offset,bytes=len(excerpt),
        sha256=hashlib.sha256(excerpt).hexdigest(),encoding='Base64 exact terminal exception excerpt'),
    machine_readonly_context=json.loads(os.environ['C71_MACHINE_READONLY_JSON']),
    later_fit_seeds_started=False,neural_free_val_assessment_started=False,reserved_eval_rows_accessed=0,
    accepted_final_directions=0,formal_training_currently_running=False,
    next_action='Bounded checkpoint800 replay to distinguish a reproducible step failure from an intermittent runtime fault; no automatic long rerun.')
dump(OUT/'RUNTIME_INCIDENT71_V2.json',result)
print(json.dumps({k:result[k] for k in ['status','first_fault_module','second_fault_module','last_checkpoint','checkpoint_tensor_audit','annealing_temperature_exact_replay','machine_readonly_context']}),flush=True)
