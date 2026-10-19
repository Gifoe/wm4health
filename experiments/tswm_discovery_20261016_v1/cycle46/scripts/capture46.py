"""Durable private supervisor; own processes only, no global environment changes."""
import pathlib,subprocess,sys,os,json,datetime,concurrent.futures,traceback
R=pathlib.Path(__file__).resolve().parents[1];O=R/'outputs';O.mkdir(exist_ok=True);P=O/'supervisor_v1.json';assert not P.exists()
def save(x):P.write_text(json.dumps(x,indent=2),encoding='utf-8')
def run(script,args,logname):
 cmd=[sys.executable,'-B','-u',str(R/'scripts'/script),*args]
 with (O/logname).open('w',encoding='utf-8') as log:
  task=subprocess.Popen(cmd,cwd=R,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0);code=task.wait()
 if code:raise RuntimeError(f'{script} {args} failed exit={code}; see {logname}')
 return {'script':script,'arguments':args,'exit_code':code,'pid':task.pid,'log':logname}
state={'status':'running','pid':os.getpid(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':[]};save(state)
try:
 for script in ['data46.py','qualify46.py']:
  row=run(script,[],script+'.log');state['completed'].append(row);save(state)
 qual=json.loads((O/'qualification_v1/qualification.json').read_text(encoding='utf-8'));workers=2 if qual['selected_training_device']=='cpu' else 1;state['training_workers']=workers;state['stage']='formal_training';save(state)
 with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
  jobs={}
  for seed in [2026461,2026462]:
   for obs,dyn in [(0,0),(0,1),(1,0),(1,1)]:
    args=['--obs',str(obs),'--dyn',str(dyn),'--seed',str(seed)];f=pool.submit(run,'train46.py',args,f'train_o{obs}d{dyn}_seed{seed}.log');jobs[f]=(obs,dyn,seed)
  for f in concurrent.futures.as_completed(jobs):state['completed'].append(f.result());save(state)
 state['status']='FITS_COMPLETE_EVAL_UNOPENED';state['completed_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save(state)
except Exception:
 state.update(status='failed',traceback=traceback.format_exc());save(state);raise
