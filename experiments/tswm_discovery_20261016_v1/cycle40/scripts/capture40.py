"""Guarded hidden supervisor; one initial qualification then one baseline screen."""
import pathlib,sys,subprocess,os,json,datetime
R=pathlib.Path(__file__).resolve().parents[1];O=R/'outputs';O.mkdir(exist_ok=True)
assert not (O/'supervisor_v1.json').exists() and not (O/'screen_v1').exists()
record={'supervisor_pid':os.getpid(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'running','stages':[]}
path=O/'supervisor_v1.json'
for stage in ['qualify40.py','screen40.py']:
 with (O/(stage+'.stdout.log')).open('xb') as stdout,(O/(stage+'.stderr.log')).open('xb') as stderr:
  p=subprocess.Popen([sys.executable,'-B','-u',str(R/'scripts'/stage)],stdout=stdout,stderr=stderr,cwd=str(R),creationflags=subprocess.CREATE_NO_WINDOW)
  row={'stage':stage,'worker_launcher_pid':p.pid,'status':'running'};record['stages'].append(row);path.write_text(json.dumps(record,indent=2))
  code=p.wait();row.update(status='completed' if code==0 else 'failed',exit_code=code);path.write_text(json.dumps(record,indent=2))
 if code:break
record.update(status='completed' if code==0 else 'failed',exit_code=code,ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());path.write_text(json.dumps(record,indent=2));sys.exit(code)
