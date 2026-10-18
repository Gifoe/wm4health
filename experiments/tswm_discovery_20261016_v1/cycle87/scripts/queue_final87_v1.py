"""Wait on the actual evaluation queue, verify closed results, prepare QA snapshot."""
from common87_v1 import *
import ctypes,subprocess
from ctypes import wintypes
K=ctypes.WinDLL('kernel32',use_last_error=True);K.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];K.OpenProcess.restype=wintypes.HANDLE;K.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD];K.WaitForSingleObject.restype=wintypes.DWORD;K.CloseHandle.argtypes=[wintypes.HANDLE]
def main():
 pid=read(O/'QUEUE_HANDLE87_V2.json')['queue_pid'];exit_path=O/'QUEUE_EXIT87_V2.json';handle=K.OpenProcess(0x100000|0x1000,False,pid)
 assert handle or exit_path.exists(),'Missing queue handle and terminal state; inspect instead of restarting'
 write(O/'FINAL_QUEUE_HANDLE87_V1.json',dict(utc=utc(),pid=os.getpid(),captured_evaluation_queue_pid=pid,live_handle_confirmed=bool(handle)))
 try:
  while handle and K.WaitForSingleObject(handle,15000)==258:
   tmp=O/'FINAL_QUEUE_PROGRESS87_V1.tmp';tmp.write_text(json.dumps(dict(utc=utc(),status='CONFIRMED_EVALUATION_QUEUE_HANDLE_LIVE',pid=pid)),encoding='utf-8');tmp.replace(O/'FINAL_QUEUE_PROGRESS87_V1.json')
 finally:
  if handle:K.CloseHandle(handle)
 assert exit_path.exists() and read(exit_path)['evaluation_exit_code']==0,'Evaluation did not close successfully: verification withheld'
 code=subprocess.call([sys.executable,'-B','-u',str(C/'scripts/capture87_v1.py'),'verify']);assert code==0,'Verifier failed; retain receipts and inspect'
 code=subprocess.call([sys.executable,'-B','-u',str(C/'scripts/finalize87_v2.py')]);write(O/'FINAL_QUEUE_EXIT87_V1.json',dict(utc=utc(),exit_code=code,status='SNAPSHOT_PENDING_MANUAL_FIGURE_QA' if code==0 else 'FINALIZATION_FAILED'));assert code==0
if __name__=='__main__':main()
