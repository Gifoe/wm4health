"""Wait on the specific fit supervisor, then diagnose/score/verify once."""
import pathlib,sys,os,subprocess,time,json,datetime
R=pathlib.Path(__file__).resolve().parents[1];O=R/'outputs';path=O/'evaluation_supervisor_v1.json';assert not path.exists();state={'status':'waiting_for_all_fits','pid':os.getpid(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':[]}
def save():path.write_text(json.dumps(state,indent=2),encoding='utf-8')
save();start=time.monotonic()
while True:
 p=O/'supervisor_v1.json'
 if p.exists():
  try:s=json.loads(p.read_text())
  except json.JSONDecodeError:time.sleep(2);continue
  if s['status']=='ALL_SIXTEEN_FITS_COMPLETE_EVAL_UNOPENED':break
  if s['status'].startswith('FAILED'):state['status']='FIT_FAILURE_NO_EVAL';save();sys.exit(1)
 if time.monotonic()-start>10800:state['status']='WAIT_LIMIT_NO_RESTART';save();sys.exit(2)
 time.sleep(10)
for script in ['diagnose48.py','score48_v2.py','verify48_v2.py']:
 state['status']='executing_'+script;save()
 with (O/(script+'.log')).open('w',encoding='utf-8') as log:
  p=subprocess.Popen([str(R.parent/'.venv/Scripts/python.exe'),'-B','-u',str(R/'scripts'/script)],cwd=R,stdout=log,stderr=subprocess.STDOUT,creationflags=0x08000000);code=p.wait()
 state['completed'].append({'script':script,'pid':p.pid,'exit_code':code});save()
 if code:state['status']='FAILED_'+script;save();sys.exit(code)
state['status']='FULL_EVALUATION_AND_INDEPENDENT_VERIFICATION_COMPLETE';state['completed_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
