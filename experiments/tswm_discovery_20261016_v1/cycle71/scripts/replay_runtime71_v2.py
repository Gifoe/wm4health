"""Two diagnostic epochs with an explicitly separate terminal diagnostic snapshot."""
import sys,pathlib
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from runtime_affinity71_v1 import configure
mode=sys.argv[1];assert mode in ('unrestricted','efficient')
affinity=configure(mode)
from resume_native71_v2 import *
incident=read(OUT/'RUNTIME_INCIDENT71_V2.json')
manifest=read(OUT/'formal_seed2026711_resumed_v2_checkpoints.json')
assert manifest['checkpoints'][-1]==incident['last_checkpoint']
assert manifest['checkpoints'][-1]['epoch']==800
tag='runtime_replay71_v2_'+mode
bg=resumed_fit(2026711,2000,2048,tag,manifest,float.fromhex(incident['annealing_temperature_exact_replay']['float64_hex']),finish=803)
assert bg.step==802 and bg.module_updates71==bg.proposal_updates71==12848
# The original save predicate uses self.optimization_steps=2000, which is retained.
# A qualification snapshot is additive; it never enters formal checkpoint selection.
if bg.receipts71[-1]['epoch']!=802:bg.save_train_state()
last=bg.receipts71[-1];assert last['epoch']==802
obj=dict(utc=utc(),status='BOUNDED_TWO_EPOCH_RUNTIME_REPLAY_COMPLETE',mode=mode,affinity=affinity,
         parent_checkpoint=incident['last_checkpoint'],last_checkpoint=last,epochs_replayed=[801,802],
         added_module_updates=32,added_proposal_updates=32,entry_adapter=bg.resume_adapter71,
         explicit_terminal_diagnostic_snapshot=True,source_optimization_steps_remains_2000=True,
         cuda_launch_blocking=os.environ.get('CUDA_LAUNCH_BLOCKING'),
         qualification_only=True,formal_fit_complete=False,performance_evidence=False,
         reserved_eval_rows_accessed=0,root_cause_verified=False,accepted_final_directions=0)
dump(OUT/(tag+'_result.json'),obj)
print(json.dumps({k:obj[k] for k in ['status','mode','epochs_replayed','added_module_updates','qualification_only']}),flush=True)
