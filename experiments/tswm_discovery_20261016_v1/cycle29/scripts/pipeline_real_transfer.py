"""Wait on the existing live scientific pipeline, then execute the frozen measured audit."""
import runtime_profile
import datetime,hashlib,json,os,pathlib,subprocess,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/real_pipeline_v1';OWNED=False
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(state):
    p=OUT/'state.json.tmp';p.write_text(json.dumps(state,indent=2),encoding='utf-8');p.replace(OUT/'state.json')
def main():
    global OWNED
    assert not OUT.exists();OUT.mkdir(parents=True);OWNED=True
    paths=[ROOT/'REAL_TRANSFER_PLAN.md',ROOT/'outputs/real_splits/manifest.json']+[ROOT/'scripts'/name for name in ['prepare_real_splits.py','evaluate_real_transfer.py','verify_real_transfer.py','report_real_transfer.py','pipeline_real_transfer.py','world_models.py','world_models_v2.py','train_world_comparison_v2.py','evaluate_world_comparison_v2.py','verify_world_comparison_v2.py']]
    locked={p.relative_to(ROOT).as_posix():sha(p) for p in paths};(OUT/'source_lock.json').write_text(json.dumps(locked,indent=2),encoding='utf-8')
    state=dict(stage='waiting_for_main_scientific_pipeline',pid=os.getpid(),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save(state)
    import psutil
    while True:
        mainstate=read(ROOT/'outputs/pipeline_v2b/state.json');job=read(ROOT/'outputs/jobs/world_pipeline_v2b/status.json')
        assert mainstate['stage']!='failed',mainstate
        if mainstate['stage']=='completed':break
        if job['status']!='running':
            mainstate=read(ROOT/'outputs/pipeline_v2b/state.json');assert job['status']=='completed' and mainstate['stage']=='completed',(job,mainstate);break
        process=psutil.Process(job['pid']);assert process.is_running() and 'pipeline_world_v2b.py' in ' '.join(process.cmdline())
        state.update(checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),verified_main_pipeline_pid=job['pid'],main_stage=mainstate['stage']);save(state);time.sleep(20)
    for relative,digest in locked.items():assert sha(ROOT/relative)==digest,relative
    for phase,name in [('real_evaluation','capture_real_transfer.py'),('independent_real_verification','verify_real_transfer.py'),('real_report','report_real_transfer.py')]:
        state.update(stage=phase,stage_started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save(state)
        with (OUT/(phase+'_stdout.log')).open('wb') as stdout,(OUT/(phase+'_stderr.log')).open('wb') as stderr:
            r=subprocess.run([sys.executable,'-B','-u',str(ROOT/'scripts'/name)],cwd=str(ROOT),stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
        assert r.returncode==0,(phase,r.returncode)
    state.update(stage='completed',finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),accepted_directions=0,goal='ACTIVE_UNACHIEVED');save(state);print(json.dumps(state),flush=True)
if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if OWNED:
            try:state=read(OUT/'state.json')
            except Exception:state={}
            state.update(stage='failed',error=repr(exc),finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save(state)
        raise
