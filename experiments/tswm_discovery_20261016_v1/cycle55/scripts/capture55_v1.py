"""Durable, single-owner numerical audit; preserve any failure before analysis."""
from common55 import *
import subprocess,traceback
def state(status,**kw):
    p=OUT/'state55.json';p.write_text(json.dumps({'utc':utc(),'pid':os.getpid(),'status':status,**kw},indent=2),encoding='utf-8')
try:
    for name in ['qualify55_v1.py','run_population55_v1.py']:
        state('RUNNING',script=name)
        with (OUT/(name+'.stdout.log')).open('xb') as stdout,(OUT/(name+'.stderr.log')).open('xb') as stderr:
            subprocess.run([sys.executable,'-B','-u',str(ROOT/'scripts'/name)],stdout=stdout,stderr=stderr,check=True)
    state('COMPLETE_SCALAR_POPULATION_NUMERICS_PENDING_INTERPRETATION',native_training_updates=0,accepted_final_directions=0)
except Exception:
    state('FAILED_PRESERVED',traceback=traceback.format_exc());raise
