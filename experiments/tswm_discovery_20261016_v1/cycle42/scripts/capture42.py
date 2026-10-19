"""Private hidden supervised sequential experiment; no credentials in files."""
import sys,pathlib,subprocess,json,datetime,traceback
R=pathlib.Path(__file__).resolve().parents[1];O=R/'outputs';O.mkdir(parents=True,exist_ok=True)
status=O/'supervisor_v1.json';assert not status.exists()
def save(x):status.write_text(json.dumps(x,indent=2),encoding='utf-8')
record={'status':'running','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scripts':[]};save(record)
try:
 for name in ['qualify42.py','train42.py']:
  log=O/(name+'.log')
  with log.open('wb') as f:
   child=subprocess.Popen([sys.executable,'-B','-u',str(R/'scripts'/name)],cwd=R,stdout=f,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));record['scripts'].append({'script':name,'pid':child.pid,'status':'running'});save(record);code=child.wait()
  record['scripts'][-1].update(status='completed' if code==0 else 'failed',exit_code=code);save(record)
  if code:raise RuntimeError(f'{name} exit {code}; see bounded log')
 record.update(status='completed',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save(record)
except Exception:
 record.update(status='failed',traceback=traceback.format_exc(),ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save(record);raise
