import sys,pathlib,subprocess,json,datetime,hashlib
root=pathlib.Path(__file__).resolve().parents[1];out=root/'outputs';out.mkdir(parents=True,exist_ok=True);log=out/'run81_v1_stdout.log'
with log.open('xb') as f:
 p=subprocess.Popen([sys.executable,'-I','-B','-u',str(root/'scripts/run81_v1.py')],stdout=f,stderr=subprocess.STDOUT,cwd=root)
 with (out/'run81_v1_launch.json').open('x') as g:json.dump(dict(pid=p.pid,utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),g)
 code=p.wait()
with (out/'run81_v1_exit.json').open('x') as g:json.dump(dict(pid=p.pid,exit_code=code,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),log_sha256=hashlib.sha256(log.read_bytes()).hexdigest()),g,indent=2)
assert code==0,code
