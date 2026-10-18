"""Capture both bounded diagnostics and compare the complete numerical state."""
import sys,pathlib,subprocess,hashlib,json,datetime,os
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs'
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
for mode in ('unrestricted','efficient'):
    stem='runtime_replay71_v2_'+mode;log=OUT/(stem+'_stdout.log')
    with log.open('xb') as f:
        child=subprocess.Popen([sys.executable,'-I','-B','-X','faulthandler','-u',str(ROOT/'scripts'/'replay_runtime71_v2.py'),mode],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT)
        with (OUT/(stem+'_launch.json')).open('x') as g:json.dump(dict(pid=child.pid,utc=utc(),runtime_overrides=dict(CUDA_LAUNCH_BLOCKING=os.environ.get('CUDA_LAUNCH_BLOCKING'))),g)
        code=child.wait()
    with (OUT/(stem+'_exit.json')).open('x') as g:json.dump(dict(pid=child.pid,exit_code=code,utc=utc(),log_sha256=hashlib.sha256(log.read_bytes()).hexdigest()),g,indent=2)
    assert code==0,('Bounded replay failed; no formal rerun',mode,code)
sys.path.insert(0,str(ROOT/'scripts'))
from resume_native71_v2 import nested_compare,torch,read,dump,sha
torch.set_num_threads(1)
results={m:read(OUT/('runtime_replay71_v2_'+m+'_result.json')) for m in ('unrestricted','efficient')}
states={m:torch.load(r['last_checkpoint']['path'],map_location='cpu',weights_only=False) for m,r in results.items()}
keys=['module_state','proposal_state','module_optimizer','proposal_optimizer','scheduler','torch_rng','cuda_rng','numpy_rng','python_rng','train_shuffle','val_shuffle','train_structures','val_structures','SA_running_acc_rate','SA_running_factor','metrics','module_updates','proposal_updates']
comp=nested_compare({k:states['unrestricted'][k] for k in keys},{k:states['efficient'][k] for k in keys})
assert comp['max_abs']==0 and comp['non_tensor_mismatches']==0,comp
obj=dict(utc=utc(),status='BOUNDED_RUNTIME_REPLAY_AND_AFFINITY_NUMERICAL_EQUIVALENCE_PASS',
         comparison=comp,keys_compared=keys,epochs_replayed=[801,802],
         input_checkpoint=results['unrestricted']['parent_checkpoint'],
         result_receipts_sha256={m:sha(OUT/('runtime_replay71_v2_'+m+'_result.json')) for m in results},
         efficient_process_control=results['efficient']['affinity'],
         deterministic_failure_at_epoch801_not_reproduced=True,
         root_cause_verified=False,long_run_stability_verified=False,formal_training_restarted=False,
         qualification_only=True,performance_evidence=False,reserved_eval_rows_accessed=0,
         accepted_final_directions=0)
dump(OUT/'RUNTIME_REPLAY_QUALIFICATION71_V2.json',obj)
print(json.dumps(obj),flush=True)
