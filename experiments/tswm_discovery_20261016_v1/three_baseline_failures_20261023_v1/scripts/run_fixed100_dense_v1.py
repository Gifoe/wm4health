"""Sequential closure of the dense audit promised by the matched-cap protocol."""
import pathlib,sys,subprocess,json,datetime,hashlib
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/fixed100_dense_controller_v1';OUT.mkdir(exist_ok=False)
rows=[]
for mode in ('latent','raw'):
 log=OUT/(mode+'_stdout.log')
 with log.open('xb') as f:
  p=subprocess.Popen([sys.executable,'-I','-B','-X','faulthandler','-u',str(ROOT/'scripts/dense_fixed100_v1.py'),mode],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT)
  with (OUT/(mode+'_launch.json')).open('x') as g:json.dump(dict(mode=mode,pid=p.pid,utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),g)
  code=p.wait()
 row=dict(mode=mode,exit_code=code,log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 with (OUT/(mode+'_exit.json')).open('x') as g:json.dump(row,g,indent=2)
 print(json.dumps(row),flush=True);rows.append(row)
 if code:break
ok=len(rows)==2 and all(x['exit_code']==0 for x in rows)
with (OUT/'RESULTS_V1.json').open('x') as f:json.dump(dict(status='TWELVE_FIXED100_DENSE_AUDITS_CLOSED' if ok else 'DENSE_AUDIT_ERROR_RETAINED',rows=rows),f,indent=2)
sys.exit(0 if ok else 1)
