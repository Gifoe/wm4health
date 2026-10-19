"""One-shot owned queue: original timing must finish, then CPU analyses and control."""
import runtime_profile
import datetime,json,os,pathlib,subprocess,sys,time
import psutil
from world_data import ROOT,sha,dump
OUT=ROOT/'outputs/fixed_update_pipeline_v1';OWNED=False

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(state):
    temp=OUT/'state.json.tmp';dump(temp,state);temp.replace(OUT/'state.json')
def main():
    global OWNED
    assert not OUT.exists();OUT.mkdir();OWNED=True
    names=['analyze_world_attribution.py','analyze_finite_plan_utility.py','verify_secondary_analyses.py','train_fixed_update_control.py','evaluate_fixed_update_control.py','verify_fixed_update_control.py','pipeline_fixed_update_control.py','world_models.py','world_models_v2.py','world_data.py','train_world_comparison_v2.py','evaluate_world_comparison_v2.py','verify_world_comparison_v2.py']
    paths=[ROOT/name for name in ['FIXED_UPDATE_CONTROL_PLAN.md','RESULT_ATTRIBUTION_PLAN.md','FINITE_PLAN_UTILITY_PLAN.md']]+[ROOT/'scripts'/name for name in names]
    lock={p.relative_to(ROOT).as_posix():sha(p) for p in paths};dump(OUT/'source_lock.json',lock)
    state=dict(stage='waiting_for_original_timing',pipeline_pid=os.getpid(),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());write(state)
    while True:
        job=read(ROOT/'outputs/jobs/inference_controls_v1/status.json')
        if job['status']!='running':
            assert job['status']=='completed' and job['exit_code']==0,job
            assert read(ROOT/'outputs/inference_controls_pipeline_v1/state.json')['stage']=='completed';break
        try:
            process=psutil.Process(job['pid']);assert process.is_running() and 'pipeline_inference_controls.py' in ' '.join(process.cmdline())
        except psutil.NoSuchProcess:time.sleep(2);continue
        state.update(verified_original_timing_pid=job['pid'],checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());write(state);time.sleep(20)
    for name,digest in lock.items():assert sha(ROOT/name)==digest,name
    stages=[('world_attribution','analyze_world_attribution.py'),('finite_plan_utility','analyze_finite_plan_utility.py'),('secondary_verification','verify_secondary_analyses.py'),('fixed_training','train_fixed_update_control.py'),('fixed_evaluation','evaluate_fixed_update_control.py'),('fixed_verification','verify_fixed_update_control.py')]
    for stage,name in stages:
        state.update(stage=stage,stage_started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());write(state)
        with (OUT/(stage+'_stdout.log')).open('wb') as stdout,(OUT/(stage+'_stderr.log')).open('wb') as stderr:
            run=subprocess.run([sys.executable,'-B','-u',str(ROOT/'scripts'/name)],cwd=str(ROOT),stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
        assert run.returncode==0,(stage,run.returncode)
    state.update(stage='completed',finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),accepted_final_directions=0,goal='ACTIVE_UNACHIEVED');write(state);print(json.dumps(state),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if OWNED:
            try:state=read(OUT/'state.json')
            except (FileNotFoundError,json.JSONDecodeError):state={}
            state.update(stage='failed',error=repr(exc));write(state)
        raise
