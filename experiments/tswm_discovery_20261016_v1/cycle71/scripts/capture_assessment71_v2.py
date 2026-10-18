import pathlib,subprocess,sys,json,datetime,hashlib
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs';mode=sys.argv[1];assert mode in ('queue','eval')
stem='queue_free_simulation71_v2' if mode=='queue' else 'free_simulation71_v2'
script='queue_free_simulation71_v2.py' if mode=='queue' else 'evaluate_free_simulation71_v2.py'
log=OUT/(stem+'_stdout.log')
with log.open('xb') as f:
 child=subprocess.Popen([sys.executable,'-I','-B','-X','faulthandler','-u',str(ROOT/'scripts'/script)],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT)
 with (OUT/(stem+'_launch.json')).open('x') as g:json.dump(dict(pid=child.pid,utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),g)
 code=child.wait()
with (OUT/(stem+'_exit.json')).open('x') as g:json.dump(dict(pid=child.pid,exit_code=code,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),log_sha256=hashlib.sha256(log.read_bytes()).hexdigest()),g,indent=2)
assert code==0,code
