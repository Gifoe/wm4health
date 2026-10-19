"""Add independent raw-data parser before ordinary result closure."""
import pathlib,sys,subprocess,json,os,datetime
R=pathlib.Path(__file__).resolve().parents[1];O=R/'outputs';assert (O/'screen_v1/COMPLETE.json').exists();assert not (O/'closure_worker_v3.json').exists()
out={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),'stages':[]}
for file in ['verify_raw40_v2.py','verify40.py','check_information40.py','analyze40.py']:
 q=subprocess.run([sys.executable,'-B','-u',str(R/'scripts'/file)],cwd=str(R),stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
 (O/(file+'.closure_v3.log')).write_bytes(q.stdout);out['stages'].append({'script':file,'exit_code':q.returncode});(O/'closure_worker_v3.json').write_text(json.dumps(out,indent=2))
 if q.returncode:sys.exit(q.returncode)
out.update(status='completed',ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());(O/'closure_worker_v3.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
