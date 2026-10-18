from common88_v1 import *
import subprocess,time

def main():
    run=O/'plan_campaign_v1';run.mkdir(parents=True,exist_ok=True)
    handle=run/'HANDLE.json';assert not handle.exists(),'Inspect existing process before restarting'
    write(handle,dict(pid=os.getpid(),started_utc=utc(),argv=sys.argv))
    for arm in ['stock','lagged','acrkn']:
        out=run/f'{arm}.log';assert not out.exists()
        command=[sys.executable,'-B','-X','faulthandler','-u',str(C/'scripts/train_plan88_v1.py'),'--arm',arm]
        with out.open('xb') as f:
            child=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,cwd=str(ROOT),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            write(run/f'{arm}_HANDLE.json',dict(pid=child.pid,utc=utc(),command=command))
            rc=child.wait()
        write(run/f'{arm}_EXIT.json',dict(returncode=rc,utc=utc()))
        if rc:raise RuntimeError((arm,rc))
    write(run/'EXIT.json',dict(returncode=0,utc=utc(),fits=3))

if __name__=='__main__':main()
