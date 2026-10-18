import subprocess,sys,os,json,datetime
from pathlib import Path
C=Path(__file__).resolve().parents[1];stage=sys.argv[1];assert stage in ['generate','train','evaluate','verify']
O=C/'outputs'/(stage+'_v1');O.mkdir(parents=True,exist_ok=True)
def record(obj):return dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**obj)
with (O/'worker87_v1.log').open('xb') as f:
 p=subprocess.Popen([sys.executable,'-B','-X','faulthandler','-u',str(C/'scripts'/(stage+'87_v1.py'))],stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2'))
 with (O/'WORKER_HANDLE87_V1.json').open('x') as g:json.dump(record(dict(capture_pid=os.getpid(),worker_pid=p.pid,stage=stage)),g)
 code=p.wait()
with (O/'WORKER_EXIT87_V1.json').open('x') as f:json.dump(record(dict(capture_pid=os.getpid(),worker_pid=p.pid,exit_code=code,stage=stage)),f)
sys.exit(code)
