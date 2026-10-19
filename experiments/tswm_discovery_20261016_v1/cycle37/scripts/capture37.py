"""One CIM-owned background worker, logs retained; no global services."""
import pathlib,sys,os,subprocess,json,datetime
R=pathlib.Path(__file__).resolve().parents[1];O=R/'outputs'
assert not (O/'learning_v2').exists() and not (O/'supervisor_v2.json').exists(),'Duplicate/partial worker refused'
rec={'supervisor_pid':os.getpid(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
with (O/'training_stdout_v2.log').open('xb') as stdout,(O/'training_stderr_v2.log').open('xb') as stderr:
    p=subprocess.Popen([sys.executable,'-B','-u',str(R/'scripts/run_learning37_v2.py')],
        cwd=str(R),stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
    rec.update(status='running',worker_launcher_pid=p.pid)
    (O/'supervisor_v2.json').write_text(json.dumps(rec,indent=2),encoding='utf-8')
    code=p.wait()
rec.update(status='completed' if code==0 else 'failed',exit_code=code,ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(O/'supervisor_v2.json').write_text(json.dumps(rec,indent=2),encoding='utf-8');sys.exit(code)
