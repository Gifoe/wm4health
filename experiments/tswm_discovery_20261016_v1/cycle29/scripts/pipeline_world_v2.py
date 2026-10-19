"""Continue this private experiment from the confirmed live trainer; no recurring scheduler."""
import runtime_profile
import datetime,hashlib,json,os,pathlib,subprocess,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/pipeline_v2'
OWNED=False
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(value):
    temp=OUT/'state.json.tmp';temp.write_text(json.dumps(value,indent=2),encoding='utf-8');temp.replace(OUT/'state.json')
def main():
    global OWNED
    assert not OUT.exists();OUT.mkdir(parents=True);OWNED=True
    source=[ROOT/'MODEL_COMPARISON_PLAN.md',ROOT/'NORMALIZATION_AMENDMENT.md',ROOT/'EVAL_MECHANISM_ABLATION_PLAN.md']+[ROOT/'scripts'/name for name in ['world_models.py','world_models_v2.py','world_data.py','train_world_comparison_v2.py','evaluate_world_comparison_v2.py','verify_world_comparison_v2.py','build_cycle29_reports.py']]
    locked={str(p.relative_to(ROOT)):sha(p) for p in source};(OUT/'source_lock.json').write_text(json.dumps(locked,indent=2),encoding='utf-8')
    started=datetime.datetime.now(datetime.timezone.utc).isoformat();state=dict(stage='waiting_for_live_training',pipeline_pid=os.getpid(),started_utc=started,training_job='world_training_v2',expected_training_script_sha256=locked['scripts/train_world_comparison_v2.py']);write(state)
    job=ROOT/'outputs/jobs/world_training_v2/status.json'
    import psutil
    while True:
        status=json.loads(job.read_text(encoding='utf-8'));assert status['script_sha256']==state['expected_training_script_sha256']
        if status['status']!='running':
            assert status['status']=='completed' and status['exit_code']==0,status;break
        try:
            process=psutil.Process(status['pid']);assert process.is_running() and 'train_world_comparison_v2.py' in ' '.join(process.cmdline())
        except psutil.NoSuchProcess:
            # Allow the job runner to atomically publish the terminal child status.
            time.sleep(2);status=json.loads(job.read_text());assert status['status']=='completed' and status['exit_code']==0,status;break
        state.update(checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),verified_live_pid=status['pid']);write(state);time.sleep(20)
    for relative,digest in locked.items():assert sha(ROOT/relative)==digest,relative
    for stage,name in [('evaluating','capture_world_evaluation_v2.py'),('independent_verification','verify_world_comparison_v2.py'),('building_reports','build_cycle29_reports.py')]:
        state.update(stage=stage,stage_started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());write(state)
        with (OUT/(stage+'_stdout.log')).open('wb') as stdout,(OUT/(stage+'_stderr.log')).open('wb') as stderr:
            completed=subprocess.run([sys.executable,'-B','-u',str(ROOT/'scripts'/name)],cwd=str(ROOT),stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
        assert completed.returncode==0,(stage,completed.returncode)
    state.update(stage='completed',finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),accepted_final_directions=0,goal='ACTIVE_UNACHIEVED');write(state);print(json.dumps(state),flush=True)
if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if OWNED:
            try:state=json.loads((OUT/'state.json').read_text())
            except (FileNotFoundError,json.JSONDecodeError):state={}
            state.update(stage='failed',error=repr(exc),finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());write(state)
        raise
