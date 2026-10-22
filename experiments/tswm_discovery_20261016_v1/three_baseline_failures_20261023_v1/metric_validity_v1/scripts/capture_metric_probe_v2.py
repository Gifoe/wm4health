"""Capture outside the new scientific output directory; preserve v1 before launch."""
import pathlib,sys,subprocess,json,datetime,hashlib
ROOT=pathlib.Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs';stem='metric_validity_probe_v2';log=OUT/(stem+'_stdout.log')
assert OUT.is_dir()
assert not (ROOT/'metric_validity_v1/outputs').exists()
with log.open('xb') as f:
    p=subprocess.Popen([sys.executable,'-I','-B','-X','faulthandler','-u',str(ROOT/'metric_validity_v1/scripts/run_metric_probe_v1.py')],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT)
    with (OUT/(stem+'_launch.json')).open('x') as g:json.dump(dict(pid=p.pid,utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),g)
    code=p.wait()
with (OUT/(stem+'_exit.json')).open('x') as g:json.dump(dict(pid=p.pid,exit_code=code,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),log_sha256=hashlib.sha256(log.read_bytes()).hexdigest()),g,indent=2)
sys.exit(code)
