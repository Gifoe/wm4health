from common56 import *
import subprocess,traceback
def state(status,**kw):
    (OUT/'state56_v1.json').write_text(json.dumps({'utc':utc(),'pid':os.getpid(),'status':status,**kw},indent=2),encoding='utf-8')
try:
    for name in ['qualify56_v1.py','run56_v1.py']:
        state('RUNNING',script=name)
        with (OUT/(name+'.stdout.log')).open('xb') as out,(OUT/(name+'.stderr.log')).open('xb') as err:
            subprocess.run([sys.executable,'-B','-u',str(ROOT/'scripts'/name)],stdout=out,stderr=err,check=True)
    state('COMPLETE_FINITE_REFERENCE_NUMERICS_PENDING_INTERPRETATION')
except Exception:
    state('FAILED_PRESERVED',traceback=traceback.format_exc());raise
