"""Wait on the verified scorer owner; then interpret and render, without selection."""
from common57 import *
import ctypes,subprocess,time,traceback
state_path=OUT/'close_state57_v1.json'
def state(status,**extra):
    tmp=state_path.with_suffix('.tmp')
    tmp.write_text(json.dumps({'utc':utc(),'pid':os.getpid(),'status':status,**extra},indent=2),encoding='utf-8');tmp.replace(state_path)
def stable_read(path):
    for attempt in range(5):
        try:return read(path)
        except json.JSONDecodeError:
            if attempt==4:raise
            time.sleep(.05)
def live(pid):
    fn=ctypes.windll.kernel32.OpenProcess;fn.argtypes=[ctypes.c_ulong,ctypes.c_bool,ctypes.c_ulong];fn.restype=ctypes.c_void_p
    handle=fn(0x1000,False,pid)
    if not handle:return False
    try:
        code=ctypes.c_ulong();get=ctypes.windll.kernel32.GetExitCodeProcess;get.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_ulong)]
        assert get(handle,ctypes.byref(code));return code.value==259
    finally:
        close=ctypes.windll.kernel32.CloseHandle;close.argtypes=[ctypes.c_void_p];close(handle)
try:
    previous=stable_read(OUT/'post_state57_v2.json');owner=previous['pid']
    while True:
        previous=stable_read(OUT/'post_state57_v2.json');assert previous['pid']==owner
        if previous['status']=='ALL_32_MODELS_AND_MAPS_SEALED_AND_SCORED_PENDING_INTERPRETATION':break
        assert previous['status'] in ['RUNNING','WAITING_ON_LIVE_TRAINING_OWNER'],previous
        assert live(owner),'Postprocess owner absent without terminal score receipt: preserve, never restart'
        state('WAITING_ON_LIVE_POSTPROCESS_OWNER',postprocess_pid=owner)
        time.sleep(20)
    for name in ['analyze57_v1.py','figures57_v1.py']:
        with (OUT/(name+'.stdout.log')).open('xb') as out,(OUT/(name+'.stderr.log')).open('xb') as err:
            child=subprocess.Popen([sys.executable,'-B','-u',str(ROOT/'scripts'/name)],stdout=out,stderr=err,cwd=ROOT)
            state('RUNNING',script=name,child_pid=child.pid)
            assert child.wait()==0,name
    state('INTERPRETATION_AND_FIGURES_COMPLETE_ACTUAL_VISUAL_QA_PENDING')
except Exception:
    state('FAILED_PRESERVED',traceback=traceback.format_exc());raise
