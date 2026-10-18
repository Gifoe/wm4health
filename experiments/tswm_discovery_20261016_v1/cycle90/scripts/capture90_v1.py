from common90_v1 import *
import subprocess

def run_stage(stage,seed):
    name=f'{stage}_{seed}';dest=O/name;dest.mkdir(parents=True,exist_ok=True);assert not (dest/'HANDLE.json').exists()
    cmd=[sys.executable,'-B','-X','faulthandler','-u',str(C/f'scripts/{stage}90_v1.py'),str(seed)]
    with (dest/'raw.log').open('xb') as f:
        p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,cwd=str(ROOT),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));write(dest/'HANDLE.json',dict(pid=p.pid,utc=utc(),command=cmd));rc=p.wait()
    write(dest/'EXIT.json',dict(utc=utc(),returncode=rc));assert rc==0,(name,rc)

def main():
    run=O/'campaign_v1';run.mkdir(parents=True,exist_ok=True);assert not (run/'HANDLE.json').exists();write(run/'HANDLE.json',dict(pid=os.getpid(),utc=utc(),protocol_sha256=sha(C/'PROTOCOL90_V1.md')))
    completed=[]
    try:
        for seed in [2026901,2026902,2026903]:
            run_stage('train',seed);run_stage('screen',seed);screen=read(O/f'CAL_SCREEN90_{seed}_V1.json');completed.append(seed)
            if seed==2026901 and screen['status']!='CAL_COMPETENCE_PASS':break
        write(run/'EXIT.json',dict(utc=utc(),returncode=0,completed_seeds=completed,eval_opened=False))
    except BaseException as exc:
        write(run/'EXIT.json',dict(utc=utc(),returncode=1,completed_seeds=completed,error=repr(exc),eval_opened=False));raise

if __name__=='__main__':main()
