import pathlib,subprocess,sys,json,datetime,hashlib
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs';OUT.mkdir(exist_ok=True,parents=True);log=OUT/'stage_data67_v1_stdout.log'
with log.open('xb') as out:
 child=subprocess.Popen([sys.executable,'-I','-B','-u',str(ROOT/'scripts'/'stage_data67_v1.py')],stdout=out,stderr=subprocess.STDOUT)
 with (OUT/'stage_launch67_v1.json').open('x') as f:json.dump(dict(pid=child.pid,utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),f)
 code=child.wait()
with (OUT/'stage_exit67_v1.json').open('x') as f:json.dump(dict(pid=child.pid,exit_code=code,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),log_sha256=hashlib.sha256(log.read_bytes()).hexdigest()),f,indent=2)
