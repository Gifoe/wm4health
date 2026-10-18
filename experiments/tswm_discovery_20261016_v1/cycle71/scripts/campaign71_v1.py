import sys,pathlib,subprocess
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common71_v1 import *
assert read(OUT/'qualification71_v1_exit.json')['exit_code']==0 and read(OUT/'QUALIFICATION71_V1.json')['status']=='PASS';rows=[]
for seed in [2026711,2026712,2026713]:
 stem='formal_seed'+str(seed)+'_v1';log=OUT/(stem+'_stdout.log')
 with log.open('xb') as f:
  child=subprocess.Popen([sys.executable,'-I','-B','-X','faulthandler','-u',str(ROOT/'scripts'/'train71_v1.py'),'--seed',str(seed)],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT);dump(OUT/(stem+'_launch.json'),dict(pid=child.pid,utc=utc()));print('LIVE_SEED',seed,'PID',child.pid,flush=True);code=child.wait()
 dump(OUT/(stem+'_exit.json'),dict(pid=child.pid,exit_code=code,utc=utc(),log_sha256=sha(log)));assert code==0,(seed,code);r=read(OUT/(stem+'_result.json'));rows.append(dict(seed=seed,result_sha256=sha(OUT/(stem+'_result.json')),module_updates=r['module_updates'],selected=r['selected'],last=r['last']));print('CLOSED_SEED',seed,'updates',r['module_updates'],flush=True)
dump(OUT/'CAMPAIGN71_V1.json',dict(utc=utc(),status='THREE_NATIVE_SUBSET_FITS_CLOSED_TRAIN_VAL_ONLY',fits=rows,module_updates=sum(x['module_updates'] for x in rows),reserved_eval_rows_accessed=0,new_method=False,accepted_final_directions=0))
