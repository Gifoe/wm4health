from common89_v1 import *
import subprocess

def main():
    run=O/'critic_stress_v1';run.mkdir(parents=True,exist_ok=True);assert not (run/'HANDLE.json').exists()
    write(run/'HANDLE.json',dict(pid=os.getpid(),utc=utc()))
    command=[sys.executable,'-B','-X','faulthandler','-u',str(C/'scripts/critic_stress89_v1.py')]
    with (run/'raw.log').open('xb') as f:
        proc=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,cwd=str(ROOT),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));write(run/'WORKER_HANDLE.json',dict(pid=proc.pid,utc=utc(),command=command));rc=proc.wait()
    write(run/'EXIT.json',dict(utc=utc(),returncode=rc));assert rc==0,rc

if __name__=='__main__':main()
