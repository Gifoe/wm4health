"""Wait on a live Windows campaign handle, then execute the frozen assessment."""
import sys,pathlib,ctypes,subprocess,time
from ctypes import wintypes
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common71_v1 import *

pid=int(read(OUT/'campaign71_v3_launch.json')['pid'])
command='Get-CimInstance Win32_Process -Filter "ProcessId='+str(pid)+'" | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress'
process=json.loads(subprocess.check_output(['powershell.exe','-NoLogo','-NoProfile','-NonInteractive','-Command',command],text=True,encoding='utf-8'))
assert process['ProcessId']==pid and '\\cycle71\\scripts\\campaign71_v3.py' in process['CommandLine'].lower()
assert not (OUT/'free_simulation71_v1_launch.json').exists()
assert read(OUT/'history_controls71_v1_exit.json')['exit_code']==0
assert read(OUT/'OPEN_LOOP_QUALIFICATION71_V1.json')['status']=='PASS_COMMON_HISTORY_FREE_ROLLOUT_INTERFACE'
k=ctypes.WinDLL('kernel32',use_last_error=True)
k.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];k.OpenProcess.restype=wintypes.HANDLE
k.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD];k.WaitForSingleObject.restype=wintypes.DWORD
k.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)];k.GetExitCodeProcess.restype=wintypes.BOOL
k.GetProcessTimes.argtypes=[wintypes.HANDLE]+[ctypes.POINTER(wintypes.FILETIME)]*4;k.GetProcessTimes.restype=wintypes.BOOL
k.CloseHandle.argtypes=[wintypes.HANDLE];k.CloseHandle.restype=wintypes.BOOL
handle=k.OpenProcess(0x100000|0x1000,False,pid)
if not handle:raise ctypes.WinError(ctypes.get_last_error())
created=wintypes.FILETIME();exited=wintypes.FILETIME();kernel=wintypes.FILETIME();user=wintypes.FILETIME()
if not k.GetProcessTimes(handle,ctypes.byref(created),ctypes.byref(exited),ctypes.byref(kernel),ctypes.byref(user)):raise ctypes.WinError(ctypes.get_last_error())
creation_ticks=(created.dwHighDateTime<<32)|created.dwLowDateTime
record=dict(utc=utc(),status='WAITING_ON_LIVE_CAMPAIGN_PROCESS_HANDLE',campaign_process=process,campaign_creation_filetime=creation_ticks,protocol_sha256=sha(ROOT/'FREE_SIMULATION_PROTOCOL71_V1.md'),evaluator_sha256=sha(ROOT/'scripts'/'evaluate_free_simulation71_v1.py'),calibration_sha256=sha(OUT/'HISTORY_CONTROL_TRAIN_CALIBRATION71_V1.json'),reserved_eval_rows_accessed=0)
dump(OUT/'FREE_SIMULATION_QUEUE71_V1.json',record);print(json.dumps(record),flush=True)
try:
 while True:
  state=k.WaitForSingleObject(handle,30000)
  if state==0:break
  if state!=258:raise ctypes.WinError(ctypes.get_last_error())
  dump(OUT/'queue_free_simulation71_v1_progress.json',dict(utc=utc(),status='CAMPAIGN_PROCESS_HANDLE_STILL_LIVE',campaign_pid=pid,campaign_creation_filetime=creation_ticks,reserved_eval_rows_accessed=0),exclusive=False)
 code=wintypes.DWORD()
 if not k.GetExitCodeProcess(handle,ctypes.byref(code)):raise ctypes.WinError(ctypes.get_last_error())
 assert code.value==0,('Campaign failed; no assessment will launch',code.value)
finally:k.CloseHandle(handle)
for _ in range(20):
 if (OUT/'campaign71_v3_exit.json').exists():break
 time.sleep(.25)
assert read(OUT/'campaign71_v3_exit.json')['exit_code']==0
assert sha(ROOT/'FREE_SIMULATION_PROTOCOL71_V1.md')==record['protocol_sha256'] and sha(ROOT/'scripts'/'evaluate_free_simulation71_v1.py')==record['evaluator_sha256']
from evaluate_free_simulation71_v1 import campaign_gate
campaign_gate()
dump(OUT/'FREE_SIMULATION_DEPENDENCY_CLOSED71_V1.json',dict(utc=utc(),status='THREE_FITS_CLOSED_LAUNCHING_FROZEN_VAL_ASSESSMENT',campaign_pid=pid,campaign_creation_filetime=creation_ticks,queue_sha256=sha(OUT/'FREE_SIMULATION_QUEUE71_V1.json'),reserved_eval_rows_accessed=0))
subprocess.run([sys.executable,'-I','-B','-X','faulthandler','-u',str(ROOT/'scripts'/'capture_assessment71_v1.py'),'eval'],check=True,cwd=ROOT)
