import pathlib,subprocess,sys,json,datetime,hashlib
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'execution_recovery_v1';OUT.mkdir(exist_ok=False);log=OUT/'stdout.log'
with log.open('xb') as f:
 p=subprocess.Popen([sys.executable,'-I','-B','-X','faulthandler','-u',str(ROOT/'scripts/run_order_probe_v3.py')],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT)
 with (OUT/'launch.json').open('x') as g:json.dump(dict(pid=p.pid,utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),g)
 code=p.wait()
with (OUT/'exit.json').open('x') as g:json.dump(dict(pid=p.pid,exit_code=code,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),log_sha256=hashlib.sha256(log.read_bytes()).hexdigest()),g,indent=2)
sys.exit(code)
