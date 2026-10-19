"""Capture full child stdout/stderr and terminal status without shell interpolation."""
import datetime,json,pathlib,subprocess,sys,os,hashlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
def run(name,script):
 assert name.replace('_','').isalnum() and pathlib.PurePath(script).name==script
 source=ROOT/'scripts'/script;assert source.is_file()
 out=ROOT/'outputs/jobs'/name;assert not out.exists();out.mkdir(parents=True)
 snapshot=source.read_bytes();(out/'script_snapshot.py').write_bytes(snapshot)
 started=datetime.datetime.now(datetime.timezone.utc).isoformat()
 command=[sys.executable,'-B','-u',str(source)]
 with (out/'stdout.log').open('wb') as stdout,(out/'stderr.log').open('wb') as stderr:
  process=subprocess.Popen(command,cwd=str(ROOT),stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
                           creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
  status={'status':'running','pid':process.pid,'supervisor_pid':os.getpid(),'started_utc':started,'command':command,'script_sha256':hashlib.sha256(snapshot).hexdigest()}
  (out/'status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
  code=process.wait()
 status.update(status='completed' if code==0 else 'failed',exit_code=code,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 temporary=out/'status.json.tmp';temporary.write_text(json.dumps(status,indent=2),encoding='utf-8');temporary.replace(out/'status.json')
 print(json.dumps(status),flush=True);return code
