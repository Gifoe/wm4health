import sys,pathlib,subprocess,json,datetime,hashlib
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs';stem='canonical74_v2';log=OUT/(stem+'_stdout.log')
with log.open('xb') as f:
 codes=[]
 for script in ['prepare74_v2.py','canonical_arx74_v2.py']:
  p=subprocess.Popen([sys.executable,'-I','-B','-u',str(ROOT/'scripts'/script)],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT)
  with (OUT/(stem+'_'+script[:-3]+'_launch.json')).open('x') as g:json.dump(dict(pid=p.pid,utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),g)
  code=p.wait();codes.append(dict(script=script,pid=p.pid,exit_code=code))
  if code:break
with (OUT/(stem+'_exit.json')).open('x') as g:json.dump(dict(processes=codes,exit_code=codes[-1]['exit_code'],utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),log_sha256=hashlib.sha256(log.read_bytes()).hexdigest()),g,indent=2)
assert all(x['exit_code']==0 for x in codes),codes
