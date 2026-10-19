"""One guarded hidden native diagnosis worker, no training or global service."""
import pathlib,sys,os,subprocess,json,datetime
R=pathlib.Path(__file__).resolve().parents[1];O=R/'outputs';O.mkdir(exist_ok=True)
assert not (O/'native_queries_v2').exists() and not (O/'supervisor_v2.json').exists()
record={'supervisor_pid':os.getpid(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
with (O/'worker_stdout_v2.log').open('xb') as stdout,(O/'worker_stderr_v2.log').open('xb') as stderr:
    p=subprocess.Popen([sys.executable,'-B','-u',str(R/'scripts/audit_queries39_v2.py')],cwd=str(R),stdout=stdout,stderr=stderr,
                       creationflags=subprocess.CREATE_NO_WINDOW)
    record.update(status='running',worker_launcher_pid=p.pid)
    (O/'supervisor_v2.json').write_text(json.dumps(record,indent=2),encoding='utf-8');code=p.wait()
record.update(status='completed' if code==0 else 'failed',exit_code=code,ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(O/'supervisor_v2.json').write_text(json.dumps(record,indent=2),encoding='utf-8');sys.exit(code)
