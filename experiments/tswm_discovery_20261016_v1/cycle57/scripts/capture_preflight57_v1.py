from common57 import *
import subprocess,traceback
def state(status,**kw):
    (OUT/'preflight_state57_v1.json').write_text(json.dumps({'utc':utc(),'pid':os.getpid(),'status':status,**kw},indent=2),encoding='utf-8')
try:
    jobs=[(PROJECT/'cycle2/.venv_rpm/Scripts/python.exe','native_fixture57_v1.py'),(pathlib.Path(sys.executable),'qualify57_v1.py'),(pathlib.Path(sys.executable),'generate57_v1.py')]
    for python,name in jobs:
        state('RUNNING',script=name)
        with (OUT/(name+'.stdout.log')).open('xb') as out,(OUT/(name+'.stderr.log')).open('xb') as err:
            subprocess.run([str(python),'-B','-u',str(ROOT/'scripts'/name)],stdout=out,stderr=err,check=True)
    state('PREFLIGHT_AND_DATA_COMPLETE_ZERO_FORMAL_UPDATES')
except Exception:
    state('FAILED_PRESERVED',traceback=traceback.format_exc());raise
