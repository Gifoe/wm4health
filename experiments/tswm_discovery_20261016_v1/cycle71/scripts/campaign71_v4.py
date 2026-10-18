"""Complete the original three-fit protocol after a verified checkpoint continuation."""
import sys,pathlib,subprocess,os
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common71_v1 import *
assert read(OUT/'RESUME_QUALIFICATION71_V2.json')['status']=='PASS_CHECKPOINT_CONTINUATION_EQUIVALENCE'
assert read(OUT/'recovery71_v2_qualify_exit.json')['exit_code']==0 and os.environ.get('CUDA_LAUNCH_BLOCKING')=='1'
assert not (OUT/'recovery71_v2_formal_launch.json').exists();rows=[]
subprocess.run([sys.executable,'-I','-B','-X','faulthandler','-u',str(ROOT/'scripts'/'capture_recovery71_v2.py'),'formal'],check=True,cwd=ROOT)
name='formal_seed2026711_resumed_v2_result.json';r=read(OUT/name);assert r['module_updates']==r['proposal_updates']==32000 and r['epochs']==2000
rows.append(dict(seed=2026711,result_file=name,result_sha256=sha(OUT/name),exit_file='recovery71_v2_formal_exit.json',module_updates=r['module_updates'],selected=r['selected'],last=r['last']))
for seed in (2026712,2026713):
 stem='formal_seed'+str(seed)+'_v3';log=OUT/(stem+'_stdout.log');assert not (OUT/(stem+'_launch.json')).exists()
 with log.open('xb') as f:
  child=subprocess.Popen([sys.executable,'-I','-B','-X','faulthandler','-u',str(ROOT/'scripts'/'train71_v3.py'),'--seed',str(seed)],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT);dump(OUT/(stem+'_launch.json'),dict(pid=child.pid,utc=utc(),runtime_overrides=dict(CUDA_LAUNCH_BLOCKING=os.environ.get('CUDA_LAUNCH_BLOCKING'))));print('LIVE_SEED',seed,'PID',child.pid,flush=True);code=child.wait()
 dump(OUT/(stem+'_exit.json'),dict(pid=child.pid,exit_code=code,utc=utc(),log_sha256=sha(log)));assert code==0,(seed,code);name=stem+'_result.json';r=read(OUT/name);assert r['module_updates']==r['proposal_updates']==32000 and r['epochs']==2000;rows.append(dict(seed=seed,result_file=name,result_sha256=sha(OUT/name),exit_file=stem+'_exit.json',module_updates=r['module_updates'],selected=r['selected'],last=r['last']));print('CLOSED_SEED',seed,flush=True)
dump(OUT/'CAMPAIGN71_V4.json',dict(utc=utc(),status='THREE_NATIVE_SUBSET_FITS_CLOSED_AFTER_DRIVER_RECOVERY_TRAIN_VAL_ONLY',fits=rows,module_updates=sum(r['module_updates'] for r in rows),update_semantics='Effective completed checkpoint lineages; discarded crash suffix is additional physical work',incident_sha256=sha(OUT/'TRAINING_INCIDENT71_V1.json'),resume_qualification_sha256=sha(OUT/'RESUME_QUALIFICATION71_V2.json'),reserved_eval_rows_accessed=0,new_method=False,accepted_final_directions=0))
