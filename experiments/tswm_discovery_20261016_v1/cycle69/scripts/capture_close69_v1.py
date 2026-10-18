import pathlib,subprocess,sys,json,datetime,hashlib
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs';stem='close69_v1';log=OUT/(stem+'_stdout.log')
with log.open('xb') as out:
 child=subprocess.Popen([sys.executable,'-I','-B','-X','faulthandler','-u',str(ROOT/'scripts'/'close69_v1.py')],stdout=out,stderr=subprocess.STDOUT,cwd=ROOT)
 with (OUT/(stem+'_launch.json')).open('x') as f:json.dump(dict(pid=child.pid,utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),f)
 code=child.wait()
with (OUT/(stem+'_exit.json')).open('x') as f:json.dump(dict(pid=child.pid,exit_code=code,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),log_sha256=hashlib.sha256(log.read_bytes()).hexdigest()),f,indent=2)
assert code==0,code
