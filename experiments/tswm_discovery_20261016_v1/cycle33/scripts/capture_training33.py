"""CIM-owned supervisor; survives the initiating SSH process ending."""
import pathlib,sys,subprocess,json,datetime,os
ROOT=pathlib.Path(__file__).resolve().parents[1];out=ROOT/'outputs'
assert not (out/'readout_fits_v1').exists(),'refusing duplicate or partial training restart'
record={'supervisor_pid':os.getpid(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
with (out/'training_stdout_cim.log').open('xb') as stdout,(out/'training_stderr_cim.log').open('xb') as stderr:
    p=subprocess.Popen([sys.executable,'-B','-u',str(ROOT/'scripts/train_readouts33.py')],stdout=stdout,stderr=stderr,
        cwd=str(ROOT),creationflags=subprocess.CREATE_NO_WINDOW)
    record.update(worker_launcher_pid=p.pid,status='running')
    (out/'supervisor.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    code=p.wait()
record.update(status='completed' if code==0 else 'failed',exit_code=code,ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(out/'supervisor.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
sys.exit(code)
