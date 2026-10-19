"""Private one-shot dependent queue; never starts another scientific training run."""
import runtime_profile
import datetime,json,os,pathlib,subprocess,sys,time
import psutil
from world_data import ROOT,sha,dump
OUT=ROOT/'outputs/inference_controls_pipeline_v1';OWNED=False

def write(state):
    temp=OUT/'state.json.tmp';dump(temp,state);temp.replace(OUT/'state.json')

def main():
    global OWNED
    assert not OUT.exists();OUT.mkdir();OWNED=True
    paths=[ROOT/'INFERENCE_IMPLEMENTATION_CONTROLS.md']+[ROOT/'scripts'/name for name in ['inference_implementations.py','audit_inference_implementations.py','world_models.py','world_models_v2.py','pipeline_inference_controls.py']]
    lock={p.relative_to(ROOT).as_posix():sha(p) for p in paths};dump(OUT/'source_lock.json',lock)
    state=dict(stage='waiting_for_real_pipeline',pipeline_pid=os.getpid(),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());write(state)
    while True:
        status=json.loads((ROOT/'outputs/jobs/real_pipeline_v1/status.json').read_text())
        if status['status']!='running':
            assert status['status']=='completed' and status['exit_code']==0,status
            assert json.loads((ROOT/'outputs/real_pipeline_v1/state.json').read_text())['stage']=='completed'
            break
        try:
            process=psutil.Process(status['pid']);assert process.is_running() and 'pipeline_real_transfer.py' in ' '.join(process.cmdline())
        except psutil.NoSuchProcess:
            time.sleep(2);continue
        state.update(verified_live_pid=status['pid'],checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());write(state);time.sleep(20)
    for name,digest in lock.items():assert sha(ROOT/name)==digest,name
    state.update(stage='equivalence_and_timing');write(state)
    with (OUT/'stdout.log').open('wb') as stdout,(OUT/'stderr.log').open('wb') as stderr:
        run=subprocess.run([sys.executable,'-B','-u',str(ROOT/'scripts/audit_inference_implementations.py')],cwd=str(ROOT),stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
    assert run.returncode==0,run.returncode
    state.update(stage='completed',finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());write(state);print(json.dumps(state),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if OWNED:
            try:state=json.loads((OUT/'state.json').read_text())
            except (FileNotFoundError,json.JSONDecodeError):state={}
            state.update(stage='failed',error=repr(exc));write(state)
        raise
