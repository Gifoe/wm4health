"""Guarded hidden experiment supervisor; no credentials in generated artifacts."""
import pathlib,sys,subprocess,json,datetime,traceback
R=pathlib.Path(__file__).resolve().parents[1];O=R/'outputs';O.mkdir(exist_ok=True);S=O/'supervisor_v1.json';assert not S.exists()
state={'status':'running','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scripts':[]}
def save():S.write_text(json.dumps(state,indent=2),encoding='utf-8')
save()
try:
 for name in ['data44.py','qualify44.py','train44.py']:
  with (O/(name+'.log')).open('wb') as log:
   child=subprocess.Popen([sys.executable,'-B','-u',str(R/'scripts'/name)],cwd=R,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));state['scripts'].append({'script':name,'pid':child.pid,'status':'running'});save();code=child.wait()
  state['scripts'][-1].update(status='completed' if code==0 else 'failed',exit_code=code);save()
  if code:raise RuntimeError(f'{name} failed, inspect preserved log')
 state.update(status='completed',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save()
except Exception:
 state.update(status='failed',traceback=traceback.format_exc());save();raise
