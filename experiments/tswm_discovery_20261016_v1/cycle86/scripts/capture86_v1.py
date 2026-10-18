import subprocess,sys,os,json,datetime
from pathlib import Path
C=Path(__file__).resolve().parents[1];O=C/'outputs'/'paired_v1';O.mkdir(parents=True,exist_ok=True)
def record(obj):return dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**obj)
with (O/'worker86_v1.log').open('xb') as f:
    p=subprocess.Popen([sys.executable,'-B','-u',str(C/'scripts'/'train86_v1.py')],stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2'))
    with (O/'WORKER_HANDLE86_V1.json').open('x') as g:json.dump(record(dict(capture_pid=os.getpid(),worker_pid=p.pid)),g)
    code=p.wait()
with (O/'WORKER_EXIT86_V1.json').open('x') as f:json.dump(record(dict(capture_pid=os.getpid(),worker_pid=p.pid,exit_code=code)),f)
sys.exit(code)
