"""Guard one native qualification; no duplicate/restart on polling timeout."""
import pathlib,sys,os,subprocess,json,datetime
R=pathlib.Path(__file__).resolve().parents[1];O=R/'outputs';O.mkdir(exist_ok=True);p=O/'supervisor_v1.json';assert not p.exists()
row={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'supervisor_pid':os.getpid(),'status':'running'}
with (O/'native_stdout_v1.log').open('xb') as a,(O/'native_stderr_v1.log').open('xb') as b:
 child=subprocess.Popen([sys.executable,'-B','-u',str(R/'scripts/qualify_lifts41.py')],stdout=a,stderr=b,cwd=str(R),creationflags=subprocess.CREATE_NO_WINDOW)
 row['worker_launcher_pid']=child.pid;p.write_text(json.dumps(row,indent=2));code=child.wait()
row.update(status='completed' if code==0 else 'failed',exit_code=code,ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());p.write_text(json.dumps(row,indent=2));sys.exit(code)
