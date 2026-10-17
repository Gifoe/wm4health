"""Wait on the verified training owner, then run sealed train maps and scoring."""
from common57 import *
import subprocess,time,traceback,ctypes
state_path=OUT/'post_state57_v2.json'
def state(status,**kw):
    state_path.write_text(json.dumps({'utc':utc(),'pid':os.getpid(),'status':status,**kw},indent=2),encoding='utf-8')
def alive(pid):
    fn=ctypes.windll.kernel32.OpenProcess;fn.argtypes=[ctypes.c_ulong,ctypes.c_bool,ctypes.c_ulong];fn.restype=ctypes.c_void_p
    handle=fn(0x1000,False,pid)
    if not handle:return False
    try:
        code=ctypes.c_ulong();get=ctypes.windll.kernel32.GetExitCodeProcess
        get.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_ulong)]
        assert get(handle,ctypes.byref(code));return code.value==259
    finally:
        close=ctypes.windll.kernel32.CloseHandle;close.argtypes=[ctypes.c_void_p];close(handle)
try:
    train=read(OUT/'training_state57_v2.json');owner=train['pid']
    while True:
        train=read(OUT/'training_state57_v2.json');assert train['pid']==owner
        if train['status']=='ALL_32_FIXED_TERMINAL_FITS_COMPLETE_HELDOUT_UNOPENED':break
        assert train['status']=='RUNNING',train
        assert alive(owner),'Training owner missing without terminal seal: preserve, do not restart'
        state('WAITING_ON_LIVE_TRAINING_OWNER',training_pid=owner,completed=len(train['completed']))
        time.sleep(20)
    for name in ['fit_maps57_v1.py','evaluate57_v1.py']:
        with (OUT/(name+'.stdout.log')).open('xb') as out,(OUT/(name+'.stderr.log')).open('xb') as err:
            child=subprocess.Popen([sys.executable,'-B','-u',str(ROOT/'scripts'/name)],stdout=out,stderr=err,cwd=ROOT)
            state('RUNNING',script=name,child_pid=child.pid)
            assert child.wait()==0,name
    state('ALL_32_MODELS_AND_MAPS_SEALED_AND_SCORED_PENDING_INTERPRETATION')
except Exception:
    state('FAILED_PRESERVED',traceback=traceback.format_exc());raise
