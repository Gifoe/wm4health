"""Time-bounded native core qualification; only kill the child tree we created."""
import pathlib,sys,subprocess,json,datetime,hashlib
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'native_core_profile_execution_recovery_v1';OUT.mkdir(exist_ok=False);log=OUT/'stdout.log'
with log.open('xb') as f:
 p=subprocess.Popen([sys.executable,'-I','-B','-u',str(ROOT/'scripts/profile_native_core_v2.py')],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT)
 with (OUT/'launch.json').open('x') as g:json.dump(dict(pid=p.pid,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),limit_seconds=600),g)
 timeout=False
 try:code=p.wait(timeout=600)
 except subprocess.TimeoutExpired:
  timeout=True;subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'],stdout=f,stderr=subprocess.STDOUT,check=False);code=p.wait()
with (OUT/'exit.json').open('x') as g:json.dump(dict(pid=p.pid,exit_code=code,timeout=timeout,limit_seconds=600,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),log_sha256=hashlib.sha256(log.read_bytes()).hexdigest()),g,indent=2)
sys.exit(code if code else (3 if timeout else 0))
