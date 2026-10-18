import pathlib,json,subprocess,sys,datetime,hashlib
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs';log=OUT/'replay66_v1_stdout.log'
with log.open('xb') as out:
 child=subprocess.Popen([sys.executable,'-I','-B','-X','faulthandler','-u',str(ROOT/'scripts'/'replay66_v1.py')],stdout=out,stderr=subprocess.STDOUT,cwd=ROOT)
 with (OUT/'replay_launch66_v1.json').open('x') as f:json.dump(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=child.pid),f)
 code=child.wait()
with (OUT/'replay_exit66_v1.json').open('x') as f:json.dump(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=child.pid,exit_code=code,log_sha256=hashlib.sha256(log.read_bytes()).hexdigest()),f,indent=2)
