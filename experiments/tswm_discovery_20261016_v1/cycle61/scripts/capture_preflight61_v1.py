"""One immutable qualification attempt with raw child logs."""
from common61 import *
import subprocess,sys,traceback
statefile=OUT/'preflight_state61_v1.json'
def state(status,**kw):
 p=statefile.with_suffix('.tmp');p.write_text(json.dumps({'utc':utc(),'pid':os.getpid(),'status':status,**kw},indent=2));p.replace(statefile)
try:
 assert not statefile.exists()
 with (OUT/'qualify61_v1.stdout.log').open('xb') as out,(OUT/'qualify61_v1.stderr.log').open('xb') as err:
  p=subprocess.Popen([sys.executable,'-B','-u',str(ROOT/'scripts/qualify61_v1.py')],stdout=out,stderr=err,cwd=ROOT);state('QUALIFYING_NATIVE_EXISTING_COMPONENTS',child_pid=p.pid);code=p.wait()
 assert code==0,code
 state('NATIVE_COMPONENT_QUALIFICATION_READY')
except Exception:
 state('FAILED_PRESERVED',traceback=traceback.format_exc());raise
