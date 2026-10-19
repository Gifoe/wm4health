"""Run sixteen source-scoped fits, preserving every log/checkpoint."""
import pathlib,subprocess,sys,os,json,datetime,concurrent.futures,traceback
R=pathlib.Path(__file__).resolve().parents[1];O=R/'outputs';P=O/'supervisor_v1.json';assert not P.exists();qual=json.loads((O/'qualification_v1/qualification.json').read_text(encoding='utf-8'));assert qual['status']=='QUALIFIED_FOR_FROZEN_FINITE_GRID_SCREEN';workers=2 if qual['selected_training_device']=='cpu' else 1
state={'status':'running','pid':os.getpid(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'workers':workers,'completed':[]}
def save():P.write_text(json.dumps(state,indent=2),encoding='utf-8')
def run(case,method,seed):
 args=['--case',case,'--method',method,'--seed',str(seed)];logname=f'train_{case}_{method}_seed{seed}.log'
 with (O/logname).open('w',encoding='utf-8') as log:
  task=subprocess.Popen([sys.executable,'-B','-u',str(R/'scripts/train47.py'),*args],cwd=R,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0);code=task.wait()
 assert code==0,(case,method,seed,code);return {'case':case,'method':method,'seed':seed,'pid':task.pid,'exit_code':code,'log':logname}
save()
try:
 with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
  jobs=[pool.submit(run,case,method,seed) for seed in [2026471,2026472] for case in ['ou','bistable'] for method in ['mean_nll','full_nll','full_laplace','full_hybrid']]
  for task in concurrent.futures.as_completed(jobs):state['completed'].append(task.result());save()
 state.update(status='ALL_SIXTEEN_FITS_COMPLETE_EVAL_UNOPENED',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
except Exception:
 state.update(status='failed',traceback=traceback.format_exc());save();raise
