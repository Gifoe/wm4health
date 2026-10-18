"""Preserve terminal native-driver failure and inspect the last trusted CPU checkpoint."""
import sys,pathlib,time,subprocess
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common71_v1 import *
import numpy as np
import torch
torch.set_num_threads(1)
stem='formal_seed2026711_v3';exit_record=read(OUT/(stem+'_exit.json'));assert exit_record['exit_code']==3221225477
assert sha(OUT/(stem+'_stdout.log'))==exit_record['log_sha256']
assert read(OUT/'campaign71_v3_exit.json')['exit_code']==read(OUT/'queue_free_simulation71_v1_exit.json')['exit_code']==1
assert not (OUT/'free_simulation71_v1_launch.json').exists()
manifest=read(OUT/(stem+'_checkpoints.json'));record=manifest['checkpoints'][-1];assert record['epoch']==650 and record['module_updates']==10416 and sha(record['path'])==record['sha256']
state=torch.load(record['path'],map_location='cpu',weights_only=False);counts=dict(tensors=0,nonfinite=0)
def inspect(value):
 if isinstance(value,torch.Tensor):
  counts['tensors']+=1
  if value.is_floating_point() or value.is_complex():counts['nonfinite']+=int((~torch.isfinite(value)).sum().item())
 elif isinstance(value,dict):
  for v in value.values():inspect(v)
 elif isinstance(value,(list,tuple)):
  for v in value:inspect(v)
inspect(state);assert counts['nonfinite']==0
scalars=storage('artifact')/stem/'logs'/'seed2026711'/'scalars.jsonl';temperatures={}
for line in scalars.open(encoding='utf-8'):
 row=json.loads(line)
 if row['tag']=='SA/temp' and row['step']<=650:temperatures[row['step']]=row['value']
assert len(temperatures)==651
temp=np.exp(-9.8);trace=[]
for i in range(651):
 choices=[temp*1.1,max(temp/1.1,np.exp(-100.))];matching=[j for j,v in enumerate(choices) if float(np.log10(v))==temperatures[i]];assert len(matching)==1,(i,matching)
 temp=choices[matching[0]];trace.append(matching[0])
assert float(np.log10(temp))==state['metrics']['SA/temp']
obj=dict(utc=utc(),status='DRIVER_ACCESS_VIOLATION_TERMINAL_CHECKPOINT_FINITE',fit_exit=exit_record,campaign_exit=read(OUT/'campaign71_v3_exit.json'),queue_exit=read(OUT/'queue_free_simulation71_v1_exit.json'),windows_error_reporting_fault_module='nvcuda64.dll',windows_error_reporting_module_version='32.0.16.1656',exception_hex='0xc0000005',root_cause='Unestablished; WER identifies faulting NVIDIA driver module, not a model-loss failure',last_checkpoint=record,checkpoint_keys=list(state),checkpoint_tensor_audit=counts,annealing_temperature_exact_replay=dict(steps=651,value=float(temp),float64_hex=float(temp).hex(),trace_sha256=hashlib.sha256(bytes(trace)).hexdigest(),log_sha256=sha(scalars)),last_completed_progress=read(OUT/(stem+'_progress.json')),later_fit_seeds_started=False,neural_free_val_assessment_started=False,reserved_eval_rows_accessed=0,accepted_final_directions=0)
dump(OUT/'TRAINING_INCIDENT71_V1.json',obj)
for name in ['literature_review.md','baseline_registry.md','experimental_results.md','negative_evidence.md','hypothesis_space.md','research_ideas.md','final_research_direction.md']:
 p=ROOT/name;raw=p.read_bytes();suffix=b'\n\n## Runtime correction: C71 driver incident\n\nThe earlier running snapshot is historical. The first fit terminated with Windows access violation 0xc0000005 at 06:14 UTC after epoch651; WER names nvcuda64.dll. The remaining seeds and free-VAL assessment did not launch. TRAINING_INCIDENT71_V1.json records terminal receipts and checkpoint verification. This is an execution failure; no model-performance or novelty conclusion follows. Resume verification is required before restarting the campaign.\n'
 assert not raw.endswith(suffix)
 with p.open('ab') as f:f.write(suffix)
print(json.dumps({k:obj[k] for k in ['status','last_checkpoint','checkpoint_tensor_audit','annealing_temperature_exact_replay']}),flush=True)
