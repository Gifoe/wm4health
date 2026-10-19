"""CIM-owned capture, one launch only, no public services or unrelated jobs."""
import pathlib, sys, os, subprocess, json, datetime
ROOT=pathlib.Path(__file__).resolve().parents[1];out=ROOT/'outputs'
assert not (out/'pilot_v2').exists() and not (out/'supervisor_v2.json').exists(),'Duplicate/partial launch refused'
record={'supervisor_pid':os.getpid(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
with (out/'training_stdout_v2.log').open('xb') as stdout,(out/'training_stderr_v2.log').open('xb') as stderr:
    p=subprocess.Popen([sys.executable,'-B','-u',str(ROOT/'scripts/run_context35_v2.py')],
           stdout=stdout,stderr=stderr,cwd=str(ROOT),creationflags=subprocess.CREATE_NO_WINDOW)
    record.update(worker_launcher_pid=p.pid,status='running')
    (out/'supervisor_v2.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    code=p.wait()
record.update(status='completed' if code==0 else 'failed',exit_code=code,
              ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(out/'supervisor_v2.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
sys.exit(code)
