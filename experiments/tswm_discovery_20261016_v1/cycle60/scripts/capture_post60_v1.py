"""Wait on a verified training-owner handle, then score the sealed campaign once."""
from common60 import *
import ctypes,time,subprocess,sys,traceback
statepath=OUT/'post_state60_v1.json'
def state(status,**kw):
    p=statepath.with_suffix('.tmp');p.write_text(json.dumps({'utc':utc(),'pid':os.getpid(),'status':status,**kw},indent=2));p.replace(statepath)
def live(pid):
    op=ctypes.windll.kernel32.OpenProcess;op.argtypes=[ctypes.c_ulong,ctypes.c_bool,ctypes.c_ulong];op.restype=ctypes.c_void_p;h=op(0x1000,False,pid)
    if not h:return False
    try:
        code=ctypes.c_ulong();fn=ctypes.windll.kernel32.GetExitCodeProcess;fn.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_ulong)];assert fn(h,ctypes.byref(code));return code.value==259
    finally:
        fn=ctypes.windll.kernel32.CloseHandle;fn.argtypes=[ctypes.c_void_p];fn(h)
try:
    owner=read(OUT/'training_state60_v1.json')['pid']
    while True:
        current=read(OUT/'training_state60_v1.json');assert current['pid']==owner
        if current['status']=='ALL_SIX_ATTEMPTS_TERMINAL_MODEL_SEAL_READY_FOR_SCORING':break
        assert current['status']=='RUNNING' and live(owner),'Training not live without terminal receipt; preserve for inspection'
        state('WAITING_ON_LIVE_TRAINING_OWNER',training_pid=owner);time.sleep(15)
    with (OUT/'score60_v1.py.stdout.log').open('xb') as out,(OUT/'score60_v1.py.stderr.log').open('xb') as err:
        child=subprocess.Popen([sys.executable,'-B','-u',str(ROOT/'scripts/score60_v1.py')],stdout=out,stderr=err,cwd=ROOT);state('SCORING',child_pid=child.pid);code=child.wait()
    assert code==0,code
    state('ALL_NSSM_AND_BANK_RESULTS_READY_FOR_INTERPRETATION')
except Exception:
    state('FAILED_PRESERVED',traceback=traceback.format_exc());raise
