from common88_v1 import *
import ctypes,ctypes.wintypes as wt,subprocess

def main():
    run=O/'paired_plan_eval_v1';run.mkdir(parents=True,exist_ok=True);assert not (run/'HANDLE.json').exists()
    source=read(O/'plan_campaign_v1/HANDLE.json');pid=source['pid']
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[wt.DWORD,wt.BOOL,wt.DWORD];kernel.OpenProcess.restype=wt.HANDLE
    kernel.WaitForSingleObject.argtypes=[wt.HANDLE,wt.DWORD];kernel.WaitForSingleObject.restype=wt.DWORD
    kernel.CloseHandle.argtypes=[wt.HANDLE];kernel.CloseHandle.restype=wt.BOOL
    handle=kernel.OpenProcess(0x00100000|0x1000,False,pid)
    if handle:
        cmd=f"Get-CimInstance Win32_Process -Filter 'ProcessId={pid}' | Select-Object ProcessId,CreationDate,CommandLine | ConvertTo-Json -Compress"
        text=subprocess.check_output(['powershell','-NoProfile','-Command',cmd],text=True,encoding='utf-8',timeout=15)
        proc=json.loads(text);assert 'capture_plan88_v1.py' in proc['CommandLine'],proc
        write(run/'HANDLE.json',dict(pid=os.getpid(),utc=utc(),waiting_on_pid=pid,source_process=proc))
        try:
            while True:
                state=kernel.WaitForSingleObject(handle,10000)
                if state==0:break
                assert state==258,state
        finally:kernel.CloseHandle(handle)
    else:
        assert (O/'plan_campaign_v1/EXIT.json').exists(),'Campaign process missing without terminal receipt'
        write(run/'HANDLE.json',dict(pid=os.getpid(),utc=utc(),campaign_already_terminal=True))
    end=read(O/'plan_campaign_v1/EXIT.json');assert end['returncode']==0,end
    command=[sys.executable,'-B','-X','faulthandler','-u',str(C/'scripts/evaluate_plan88_v1.py')]
    with (run/'evaluation.log').open('xb') as f:
        child=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,cwd=str(ROOT),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        write(run/'EVALUATOR_HANDLE.json',dict(pid=child.pid,utc=utc(),command=command));rc=child.wait()
    write(run/'EXIT.json',dict(utc=utc(),returncode=rc));assert rc==0,rc

if __name__=='__main__':main()
