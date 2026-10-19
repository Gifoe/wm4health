"""Bounded two-worker supervisor for all16 frozen fresh fits."""
import pathlib,sys,os,subprocess,time,json,datetime
R=pathlib.Path(__file__).resolve().parents[1];O=R/'outputs';status=O/'supervisor_v1.json';assert not status.exists();python=str(R.parent/'.venv/Scripts/python.exe');jobs=[(case,method,seed) for seed in [2026481,2026482] for case in ['ett','jena'] for method in ['node','fixed_dde','learned_dde','gru']];running=[];completed=[];state={'status':'running','pid':os.getpid(),'workers':2,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':completed}
def save():status.write_text(json.dumps(state,indent=2),encoding='utf-8')
save()
while jobs or running:
 while jobs and len(running)<2:
  case,method,seed=jobs.pop(0);name=f'train_{case}_{method}_seed{seed}.log';log=(O/name).open('w',encoding='utf-8');process=subprocess.Popen([python,'-B','-u',str(R/'scripts/train48.py'),'--case',case,'--method',method,'--seed',str(seed)],cwd=R,stdout=log,stderr=subprocess.STDOUT,creationflags=0x08000000);running.append((process,log,{'case':case,'method':method,'seed':seed,'pid':process.pid,'log':name}));save()
 for process,log,record in list(running):
  code=process.poll()
  if code is None:continue
  log.close();running.remove((process,log,record));completed.append({**record,'exit_code':code});save()
  if code:
   state['status']='FAILED_STOPPING_OWN_FIT_WORKERS'
   for other,handle,_ in running:other.terminate();other.wait();handle.close()
   save();sys.exit(code)
 time.sleep(2)
state['status']='ALL_SIXTEEN_FITS_COMPLETE_EVAL_UNOPENED';state['completed_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
