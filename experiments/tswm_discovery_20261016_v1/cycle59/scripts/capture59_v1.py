"""Run the two data qualification steps once, preserving all logs and failures."""
from common59 import *
import subprocess,sys,traceback
state_path=OUT/'state59_v1.json'
def state(status,**kw):
    p=state_path.with_suffix('.tmp');p.write_text(json.dumps({'utc':utc(),'pid':os.getpid(),'status':status,**kw},indent=2));p.replace(state_path)
try:
    for name in ['download59_v1.py','audit59_v1.py']:
        with (OUT/(name+'.stdout.log')).open('xb') as out,(OUT/(name+'.stderr.log')).open('xb') as err:
            child=subprocess.Popen([sys.executable,'-B','-u',str(ROOT/'scripts'/name)],stdout=out,stderr=err,cwd=ROOT)
            state('RUNNING',step=name,child_pid=child.pid);code=child.wait()
        assert code==0,(name,code)
    state('DATA_AND_SOURCE_AUDIT_COMPLETE_NO_MODEL_FIT')
except Exception:
    state('FAILED_PRESERVED',traceback=traceback.format_exc());raise
