"""Run ordinary control now; score only after all frozen native selections."""
import pathlib,subprocess,sys,os,json,datetime,time,traceback
R=pathlib.Path(__file__).resolve().parents[1];O=R/'outputs';P=O/'evaluation_supervisor_v1.json';assert not P.exists()
def save(x):P.write_text(json.dumps(x,indent=2),encoding='utf-8')
def run(name):
 with (O/(name+'.log')).open('w',encoding='utf-8') as log:
  task=subprocess.Popen([sys.executable,'-B','-u',str(R/'scripts'/name)],cwd=R,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0);code=task.wait()
 assert code==0,(name,code);return {'script':name,'pid':task.pid,'exit_code':code}
state={'status':'running','pid':os.getpid(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':[]};save(state)
try:
 state['completed'].append(run('control46.py'));state['status']='waiting_for_all_fits';save(state);started=time.monotonic()
 while True:
  training=json.loads((O/'supervisor_v1.json').read_text(encoding='utf-8'))
  if training['status']=='FITS_COMPLETE_EVAL_UNOPENED':break
  assert training['status']!='failed',training
  assert time.monotonic()-started<7200,'Bounded supervisor wait exceeded; inspect training status'
  time.sleep(5)
 state['status']='scoring';save(state)
 for name in ['score46.py','verify46.py']:state['completed'].append(run(name));save(state)
 state.update(status='EVALUATION_AND_INDEPENDENT_VERIFICATION_COMPLETE',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save(state)
except Exception:
 state.update(status='failed',traceback=traceback.format_exc());save(state);raise
