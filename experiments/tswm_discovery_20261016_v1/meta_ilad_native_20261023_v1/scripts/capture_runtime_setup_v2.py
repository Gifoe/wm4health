import pathlib,sys,subprocess,json,datetime,hashlib
r=pathlib.Path(__file__).resolve().parents[1];o=r/'runtime_execution_v1';o.mkdir(exist_ok=False)
with (o/'stdout.log').open('xb') as f:
 p=subprocess.Popen([sys.executable,'-I','-B','-u',str(r/'scripts/setup_cpu_runtime_v2.py')],stdout=f,stderr=subprocess.STDOUT,cwd=r)
 with (o/'launch.json').open('x') as g:json.dump(dict(pid=p.pid,utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),g)
 code=p.wait()
with (o/'exit.json').open('x') as g:json.dump(dict(exit_code=code,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),log_sha256=hashlib.sha256((o/'stdout.log').read_bytes()).hexdigest()),g,indent=2)
sys.exit(code)
