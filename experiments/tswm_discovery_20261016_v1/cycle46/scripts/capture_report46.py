"""Durable report stage, gated on complete independent verification."""
import pathlib,subprocess,sys,os,json,datetime,time,traceback
R=pathlib.Path(__file__).resolve().parents[1];O=R/'outputs';P=O/'report_supervisor_v1.json';assert not P.exists()
state={'status':'waiting_for_evaluation','pid':os.getpid(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
def save():P.write_text(json.dumps(state,indent=2),encoding='utf-8')
save()
try:
 start=time.monotonic()
 while True:
  evaluation=json.loads((O/'evaluation_supervisor_v1.json').read_text(encoding='utf-8'))
  assert evaluation['status']!='failed',evaluation
  if evaluation['status']=='EVALUATION_AND_INDEPENDENT_VERIFICATION_COMPLETE':break
  assert time.monotonic()-start<7200,'Report wait exceeded; inspect upstream stage'
  time.sleep(5)
 state['status']='reporting';save()
 with (O/'report46.py.log').open('w',encoding='utf-8') as log:
  task=subprocess.Popen([sys.executable,'-B','-u',str(R/'scripts/report46.py')],cwd=R,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0);code=task.wait()
 assert code==0,code;state.update(status='REPORTS_COMPLETE_VISUAL_QA_PENDING',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),worker_pid=task.pid);save()
except Exception:
 state.update(status='failed',traceback=traceback.format_exc());save();raise
