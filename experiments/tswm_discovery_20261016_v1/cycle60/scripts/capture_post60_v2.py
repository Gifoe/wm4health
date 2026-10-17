"""Wait on both actual training owners; seal all attempts before unified scoring."""
from common60 import *
import ctypes,time,subprocess,sys,traceback
statefile=OUT/'post_state60_v2.json'
def state(status,**kw):
    p=statefile.with_suffix('.tmp');p.write_text(json.dumps({'utc':utc(),'pid':os.getpid(),'status':status,**kw},indent=2));p.replace(statefile)
def live(pid):
    op=ctypes.windll.kernel32.OpenProcess;op.argtypes=[ctypes.c_ulong,ctypes.c_bool,ctypes.c_ulong];op.restype=ctypes.c_void_p;h=op(0x1000,False,pid)
    if not h:return False
    try:
        code=ctypes.c_ulong();fn=ctypes.windll.kernel32.GetExitCodeProcess;fn.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_ulong)];assert fn(h,ctypes.byref(code));return code.value==259
    finally:
        fn=ctypes.windll.kernel32.CloseHandle;fn.argtypes=[ctypes.c_void_p];fn(h)
try:
    assert read(OUT/'POST_SUPERSEDED60_V1.json')['status']=='WAITING_SCORER_SUPERSEDED_BEFORE_ANY_C60_TEST_SCORE'
    assert not (OUT/'TEST_OPEN60_V1.json').exists() and not (OUT/'TEST_OPEN60_V2.json').exists()
    owners={name:read(OUT/name)['pid'] for name in ['training_state60_v1.json','gru_training_state60_v1.json']}
    terminal={'training_state60_v1.json':'ALL_SIX_ATTEMPTS_TERMINAL_MODEL_SEAL_READY_FOR_SCORING','gru_training_state60_v1.json':'ALL_SIX_GRU_ATTEMPTS_TERMINAL_MODEL_SEAL_READY_FOR_SCORING'}
    while True:
        done=[]
        for name,pid in owners.items():
            current=read(OUT/name);assert current['pid']==pid
            if current['status']==terminal[name]:done.append(name);continue
            assert current['status']=='RUNNING' and live(pid),(name,current)
        if len(done)==2:break
        state('WAITING_ON_VERIFIED_TRAINING_OWNERS',owners=owners,terminal_campaigns=done);time.sleep(15)
    a=read(OUT/'MODEL_SEAL60_V1.json');b=read(OUT/'GRU_MODEL_SEAL60_V1.json');assert len(a['models'])+len(a['failed'])==len(b['models'])+len(b['failed'])==6
    combined={'utc':utc(),'status':'ALL_TWELVE_PREDECLARED_ATTEMPTS_TERMINAL_SELECTED_MODELS_SEALED','models':a['models']+b['models'],'failed':a['failed']+b['failed'],'native_seal_sha256':sha(OUT/'MODEL_SEAL60_V1.json'),'gru_seal_sha256':sha(OUT/'GRU_MODEL_SEAL60_V1.json'),'amendment_sha256':sha(ROOT/'GRU_CONTROL60_V1.md'),'test_scored':False,'overall_goal':'ACTIVE_UNACHIEVED','accepted_final_directions':0}
    dump(OUT/'COMBINED_MODEL_SEAL60_V1.json',combined)
    with (OUT/'score60_v2.py.stdout.log').open('xb') as out,(OUT/'score60_v2.py.stderr.log').open('xb') as err:
        child=subprocess.Popen([sys.executable,'-B','-u',str(ROOT/'scripts/score60_v2.py')],stdout=out,stderr=err,cwd=ROOT);state('SCORING_ALL_SEALED_MODELS',child_pid=child.pid);code=child.wait()
    assert code==0,code
    state('ALL_NSSM_GRU_AND_BANK_RESULTS_READY_FOR_INTERPRETATION')
except Exception:
    state('FAILED_PRESERVED',traceback=traceback.format_exc());raise
