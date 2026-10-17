"""Wait on the actual native supervisor, then score once and verify retained arrays."""
from native53 import *
import ctypes
from ctypes import wintypes
ctypes.windll.kernel32.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
ctypes.windll.kernel32.OpenProcess.restype=wintypes.HANDLE
ctypes.windll.kernel32.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
ctypes.windll.kernel32.GetExitCodeProcess.restype=wintypes.BOOL
ctypes.windll.kernel32.CloseHandle.argtypes=[wintypes.HANDLE]
import subprocess
import time

state=R/'outputs/evaluation_supervisor_v2.json';assert not state.exists()
source=R/'outputs/supervisor_v7.json'
started=time.monotonic()
def save(status,**extra):dump(state,{'status':status,'pid':os.getpid(),'utc':now(),**extra})
def alive(pid):
    handle=ctypes.windll.kernel32.OpenProcess(0x1000,False,pid)
    if not handle:return False
    value=ctypes.c_ulong()
    try:
        if not ctypes.windll.kernel32.GetExitCodeProcess(handle,ctypes.byref(value)):raise OSError('Cannot query observed process')
        return value.value==259
    finally:ctypes.windll.kernel32.CloseHandle(handle)
save('WAITING_ON_NATIVE_SUPERVISOR')
while time.monotonic()-started<12*3600:
    try:fit=json.loads(source.read_text())
    except json.JSONDecodeError:time.sleep(.25);continue
    if fit['status']=='ALL_40_NATIVE_FITS_COMPLETE_TRANSFER_UNOPENED':break
    if fit['status'].startswith('FAILED'):
        save('DEPENDENCY_FAILED_NO_RESTART',dependency=fit);sys.exit(1)
    if not alive(int(fit['pid'])):
        save('DEPENDENCY_PROCESS_MISSING_NO_RESTART',observed_pid=fit['pid'],last_state=fit);sys.exit(2)
    save('WAITING_ON_VERIFIED_LIVE_NATIVE_SUPERVISOR',observed_pid=fit['pid'],completed_jobs=len(fit['completed']),active=fit.get('active'))
    time.sleep(25)
else:
    save('WAIT_LIMIT_REACHED_NO_RESTART');sys.exit(2)
env=os.environ.copy();env['PYTHONPATH']=str(P/'cycle52/runtime_deps')+os.pathsep+env.get('PYTHONPATH','');env['WANDB_MODE']='disabled'
finished=[]
for script in ['score_native53_v8.py','verify_native53.py']:
    log=R/'outputs'/(script+'.log')
    with log.open('wb') as stream:
        child=subprocess.Popen([sys.executable,'-B','-u',str(R/'scripts'/script)],cwd=R,env=env,stdout=stream,stderr=subprocess.STDOUT)
        save('RUNNING_LOCKED_EVALUATION',active={'script':script,'pid':child.pid,'log':log.name},finished=finished)
        code=child.wait()
    finished.append({'script':script,'exit_code':code,'pid':child.pid})
    if code:
        save('FAILED_EVALUATION_NO_FIT_RESTART',finished=finished);sys.exit(1)
save('NATIVE_TRANSFER_SCORING_AND_ARITHMETIC_VERIFICATION_COMPLETE_ANALYSIS_PENDING',finished=finished)

