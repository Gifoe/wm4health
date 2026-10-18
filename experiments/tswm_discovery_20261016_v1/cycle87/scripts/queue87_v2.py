"""Wait on the actual captured native-training process, then open held-out once."""
from common87_v1 import *
import ctypes,subprocess,time
from ctypes import wintypes
K=ctypes.WinDLL('kernel32',use_last_error=True)
K.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];K.OpenProcess.restype=wintypes.HANDLE
K.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD];K.WaitForSingleObject.restype=wintypes.DWORD
K.GetProcessTimes.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.FILETIME),ctypes.POINTER(wintypes.FILETIME),ctypes.POINTER(wintypes.FILETIME),ctypes.POINTER(wintypes.FILETIME)];K.GetProcessTimes.restype=wintypes.BOOL
K.CloseHandle.argtypes=[wintypes.HANDLE]
def main():
 holder=read(O/'train_v1/WORKER_HANDLE87_V1.json');pid=holder['capture_pid'];exit_path=O/'train_v1/WORKER_EXIT87_V1.json';handle=K.OpenProcess(0x100000|0x1000,False,pid)
 if not handle:assert exit_path.exists(),'Missing live handle and missing terminal record; inspect, never restart'
 creation=None
 if handle:
  create,ex,kern,user=(wintypes.FILETIME() for _ in range(4));assert K.GetProcessTimes(handle,ctypes.byref(create),ctypes.byref(ex),ctypes.byref(kern),ctypes.byref(user));creation=(create.dwHighDateTime<<32)|create.dwLowDateTime
  assert creation==read(O/'QUEUE_HANDLE87_V1.json')['training_creation_filetime'],'PID reuse detected'
 write(O/'QUEUE_HANDLE87_V2.json',dict(utc=utc(),queue_pid=os.getpid(),captured_training_pid=pid,training_creation_filetime=creation,live_handle_confirmed=bool(handle),reserved_eval_accessed=False))
 try:
  while handle and K.WaitForSingleObject(handle,15000)==258:
   progress=dict(utc=utc(),status='CONFIRMED_TRAINING_PROCESS_HANDLE_LIVE',pid=pid,creation_filetime=creation,eval_opened=False)
   tmp=O/'QUEUE_PROGRESS87_V2.tmp';tmp.write_text(json.dumps(progress),encoding='utf-8');tmp.replace(O/'QUEUE_PROGRESS87_V2.json')
 finally:
  if handle:K.CloseHandle(handle)
 # Capture writes its terminal receipt immediately before exit.
 assert exit_path.exists() and read(exit_path)['exit_code']==0,'Training is terminal without success: no held-out opening'
 assert read(O/'VAL_QUALIFICATION87_V1.json')['selected_unchanged']
 assert read(O/'DATA_VERIFICATION87_V1.json')['status'].startswith('PASS_')
 assert read(O/'SELECTION87_V1.json')['status']=='NINE_NATIVE_FITS_SELECTED_BEFORE_EVALUATION'
 assert read(O/'EMPIRICAL_SELECTION87_V1.json')['status']=='SAME_SUPERVISION_EMPIRICAL_PRIORS_SEALED_BEFORE_TEST'
 assert not (O/'EVAL_OPENING87_V1.json').exists(),'Held-out already opened; inspect before launch'
 assert not (O/'evaluate_v2/WORKER_HANDLE87_V2.json').exists(),'Evaluation already has a handle; inspect instead of duplicate'
 log(dict(stage='queue_training_closed_opening_frozen_eval',training_pid=pid))
 code=subprocess.call([sys.executable,'-B','-u',str(C/'scripts/capture_eval87_v2.py')]);write(O/'QUEUE_EXIT87_V2.json',dict(utc=utc(),evaluation_exit_code=code,queue_pid=os.getpid()));assert code==0
if __name__=='__main__':main()
