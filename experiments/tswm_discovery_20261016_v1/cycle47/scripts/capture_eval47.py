"""Run locked quality and fresh independent verifier after all sixteen fits."""
import pathlib,subprocess,sys,os,json,datetime,time,traceback
R=pathlib.Path(__file__).resolve().parents[1];O=R/'outputs';P=O/'evaluation_supervisor_v1.json';assert not P.exists();state={'status':'waiting_for_all_fits','pid':os.getpid(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':[]}
def save():P.write_text(json.dumps(state,indent=2),encoding='utf-8')
save()
try:
 start=time.monotonic()
 while True:
  training=json.loads((O/'supervisor_v1.json').read_text(encoding='utf-8'));assert training['status']!='failed',training
  if training['status']=='ALL_SIXTEEN_FITS_COMPLETE_EVAL_UNOPENED':break
  assert time.monotonic()-start<7200,'Bounded wait exceeded; inspect training'
  time.sleep(5)
 state['status']='scoring';save()
 for name in ['score47.py','verify47.py']:
  with (O/(name+'.log')).open('w',encoding='utf-8') as log:
   task=subprocess.Popen([sys.executable,'-B','-u',str(R/'scripts'/name)],cwd=R,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0);code=task.wait()
  assert code==0,(name,code);state['completed'].append({'script':name,'pid':task.pid,'exit_code':code});save()
 state.update(status='FULL_EVALUATION_AND_INDEPENDENT_VERIFICATION_COMPLETE',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
except Exception:
 state.update(status='failed',traceback=traceback.format_exc());save();raise
