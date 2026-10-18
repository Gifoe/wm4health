from common89_v1 import *
import subprocess

def main():
    run=O/'csdi_train_v1';run.mkdir(parents=True,exist_ok=True);assert not (run/'HANDLE.json').exists()
    write(run/'HANDLE.json',dict(pid=os.getpid(),utc=utc(),protocol_sha256=sha(C/'CSDI_COUPLING_PROTOCOL89_V1.md'),competence_addendum_sha256=sha(C/'CSDI_COMPETENCE_ADDENDUM89_V1.md')))
    command=[sys.executable,'-B','-X','faulthandler','-u',str(C/'scripts/train_csdi89_v1.py')]
    with (run/'raw.log').open('xb') as f:
        proc=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,cwd=str(ROOT),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        write(run/'WORKER_HANDLE.json',dict(pid=proc.pid,utc=utc(),command=command));rc=proc.wait()
    write(run/'EXIT.json',dict(utc=utc(),returncode=rc));assert rc==0,rc

if __name__=='__main__':main()
