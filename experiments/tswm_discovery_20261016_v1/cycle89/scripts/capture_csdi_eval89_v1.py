from common89_v1 import *
import subprocess,time

def main():
    run=O/'csdi_eval_v1';run.mkdir(parents=True,exist_ok=True);assert not (run/'HANDLE.json').exists()
    write(run/'HANDLE.json',dict(pid=os.getpid(),utc=utc()))
    # Queue only behind the already identified capture worker, not behind a stale state file.
    train_exit=O/'csdi_train_v1/EXIT.json'
    while not train_exit.exists():
        handle=read(O/'csdi_train_v1/WORKER_HANDLE.json')
        query=subprocess.run(['powershell','-NoProfile','-NonInteractive','-Command',f"$p=Get-CimInstance Win32_Process -Filter \"ProcessId={int(handle['pid'])}\"; if($null -eq $p){{exit 3}}; if($p.CommandLine -notmatch 'train_csdi89_v1.py'){{exit 4}}"],capture_output=True,timeout=20)
        if query.returncode!=0:
            time.sleep(2)
            if not train_exit.exists():raise RuntimeError('Training handle not live and no terminal receipt; inspect rather than restart')
            break
        time.sleep(10)
    terminal=read(train_exit);assert terminal['returncode']==0,terminal
    command=[sys.executable,'-B','-X','faulthandler','-u',str(C/'scripts/eval_csdi89_v1.py')]
    with (run/'raw.log').open('xb') as f:
        proc=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,cwd=str(ROOT),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        write(run/'WORKER_HANDLE.json',dict(pid=proc.pid,utc=utc(),command=command));rc=proc.wait()
    write(run/'EXIT.json',dict(utc=utc(),returncode=rc));assert rc==0,rc

if __name__=='__main__':main()
