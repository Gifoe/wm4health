import sys,subprocess,json,datetime,os
from pathlib import Path
C=Path(__file__).resolve().parents[1];O=C/'outputs'/'attribution_v1';O.mkdir(parents=True,exist_ok=True)
def record(obj):return json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**obj),indent=2)
with (O/'worker85_v1.log').open('xb') as f:
    p=subprocess.Popen([sys.executable,'-B','-u',str(C/'scripts'/'attribution85_v1.py')],stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2'))
    with (O/'WORKER_HANDLE85_V1.json').open('x') as g:g.write(record(dict(worker_pid=p.pid,capture_pid=os.getpid(),script='scripts/attribution85_v1.py')))
    code=p.wait()
with (O/'WORKER_EXIT85_V1.json').open('x') as f:f.write(record(dict(worker_pid=p.pid,capture_pid=os.getpid(),exit_code=code)))
sys.exit(code)
