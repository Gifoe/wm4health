"""Wait for the existing fits, then independently verify and summarize once."""
import pathlib,sys,json,time,subprocess,datetime,traceback
R=pathlib.Path(__file__).resolve().parents[1];O=R/'outputs';S=O/'finish_v1.json';assert not S.exists()
state={'status':'waiting_existing_fits','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scripts':[]}
def save():S.write_text(json.dumps(state,indent=2),encoding='utf-8')
save()
try:
 deadline=time.monotonic()+1200
 while True:
  job=json.loads((O/'supervisor_v1.json').read_text())
  if job['status']=='failed':raise RuntimeError('Existing job failed; preserve and inspect it')
  if job['status']=='completed':break
  if time.monotonic()>deadline:raise TimeoutError('Existing-fit bounded wait expired')
  time.sleep(5)
 state['status']='verifying';save()
 for name in ['verify42.py','analyze42.py']:
  with (O/(name+'.log')).open('wb') as log:
   child=subprocess.Popen([sys.executable,'-B','-u',str(R/'scripts'/name)],cwd=R,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));state['scripts'].append({'script':name,'pid':child.pid});save();code=child.wait()
  state['scripts'][-1]['exit_code']=code;save()
  if code:raise RuntimeError(f'{name} failed, see its log')
 state.update(status='completed',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
except Exception:
 state.update(status='failed',traceback=traceback.format_exc());save();raise
