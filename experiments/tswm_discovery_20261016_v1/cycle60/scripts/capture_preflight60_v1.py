"""Preserve exact preflight stdout/stderr and authoritative child/owner handles."""
from common60 import *
import subprocess,sys,traceback
statepath=OUT/'preflight_state60_v1.json'
def state(status,**kw):
    p=statepath.with_suffix('.tmp');p.write_text(json.dumps({'utc':utc(),'pid':os.getpid(),'status':status,**kw},indent=2));p.replace(statepath)
try:
    for name in ['prepare60_v1.py','qualify60_v1.py']:
        with (OUT/(name+'.stdout.log')).open('xb') as out,(OUT/(name+'.stderr.log')).open('xb') as err:
            child=subprocess.Popen([sys.executable,'-B','-u',str(ROOT/'scripts'/name)],cwd=ROOT,stdout=out,stderr=err)
            state('RUNNING',step=name,child_pid=child.pid);code=child.wait()
        assert code==0,(name,code)
    state('DATA_QUALIFIED_AND_SYNTHETIC_DEVICE_SELECTION_COMPLETE')
except Exception:
    state('FAILED_PRESERVED',traceback=traceback.format_exc());raise
