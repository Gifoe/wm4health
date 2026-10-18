import subprocess,sys,os,json,datetime
from pathlib import Path
C=Path(__file__).resolve().parents[1];stem=sys.argv[1];assert stem in ['check_rollout87_v2','verify87_v3','finalize87_v4'];O=C/'outputs'/stem;O.mkdir(parents=True,exist_ok=True)
def record(obj):return dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**obj)
with (O/'worker.log').open('xb') as f:
 p=subprocess.Popen([sys.executable,'-B','-X','faulthandler','-u',str(C/'scripts'/(stem+'.py'))],stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2'))
 with (O/'HANDLE.json').open('x') as g:json.dump(record(dict(capture_pid=os.getpid(),worker_pid=p.pid,stem=stem)),g)
 code=p.wait()
with (O/'EXIT.json').open('x') as f:json.dump(record(dict(capture_pid=os.getpid(),worker_pid=p.pid,exit_code=code,stem=stem)),f)
sys.exit(code)
