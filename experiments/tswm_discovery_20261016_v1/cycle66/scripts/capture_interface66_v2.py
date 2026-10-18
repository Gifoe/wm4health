import pathlib,json,subprocess,sys,datetime,hashlib
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs';log=OUT/'interface66_v2_stdout.log'
with log.open('xb') as out:
 child=subprocess.Popen([sys.executable,'-I','-B','-X','faulthandler','-u',str(ROOT/'scripts'/'interface66_v2.py')],stdout=out,stderr=subprocess.STDOUT,cwd=ROOT)
 with (OUT/'interface_launch66_v2.json').open('x') as f:json.dump(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=child.pid),f)
 code=child.wait()
with (OUT/'interface_exit66_v2.json').open('x') as f:json.dump(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=child.pid,exit_code=code,log_sha256=hashlib.sha256(log.read_bytes()).hexdigest()),f,indent=2)
