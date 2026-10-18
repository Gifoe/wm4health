"""Wait on one verified Windows process handle; never restart training."""
from common65 import *
import ctypes,subprocess,sys,time
pid=int(sys.argv[1]);K=ctypes.WinDLL('kernel32',use_last_error=True);K.OpenProcess.argtypes=[ctypes.c_uint32,ctypes.c_int,ctypes.c_uint32];K.OpenProcess.restype=ctypes.c_void_p;K.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_uint32];K.WaitForSingleObject.restype=ctypes.c_uint32;K.CloseHandle.argtypes=[ctypes.c_void_p];handle=K.OpenProcess(0x100000|0x1000,False,pid);assert handle,('Training PID not live; inspect terminal state before launching dependent job',pid)
state('after_train_state65_v1.json',dict(status='WAITING_ON_LIVE_TRAIN_PROCESS_HANDLE',training_pid=pid,process_handle_open=True))
try:
 while True:
  ret=K.WaitForSingleObject(handle,30000)
  if ret==0:break
  assert ret==258,ret
finally:K.CloseHandle(handle)
for attempt in range(30):
 if (OUT/'train_exit65_v1.json').exists():break
 time.sleep(1)
receipt=read(OUT/'train_exit65_v1.json');assert receipt['exit_code']==0,receipt;assert read(OUT/'FITS65_V1.json')['status']=='ALL_18_MODELS_FROZEN'
with (OUT/'evaluate65_v1_stdout.log').open('xb') as log:
 child=subprocess.Popen([sys.executable,'-B','-u',str(ROOT/'scripts'/'evaluate65_v1.py')],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT);state('after_train_state65_v1.json',dict(status='EVALUATION_CHILD_LIVE',pid=child.pid,training_pid=pid));code=child.wait()
dump(OUT/'evaluation_exit65_v1.json',dict(utc=utc(),pid=child.pid,exit_code=code,status='EVALUATION_CLOSED' if code==0 else 'EVALUATION_FAILED',log_sha256=sha(OUT/'evaluate65_v1_stdout.log')));state('after_train_state65_v1.json',dict(status='TERMINAL',exit_code=code))
