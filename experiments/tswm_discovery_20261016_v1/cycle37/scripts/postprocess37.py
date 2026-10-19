"""Wait for the original live supervisor, then run the independent verifier once."""
import pathlib,json,time,subprocess,sys,os,datetime
R=pathlib.Path(__file__).resolve().parents[1];O=R/'outputs';dest=O/'postprocess_v1.json'
assert not dest.exists(),'Duplicate postprocess refused'
def status(state,**kw):
    temp=dest.with_suffix('.tmp');temp.write_text(json.dumps({'status':state,'pid':os.getpid(),
        'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw},indent=2),encoding='utf-8');temp.replace(dest)
supervisor=O/'supervisor_v2.json';assert supervisor.exists(),'Specific original supervisor required'
status('waiting_for_original_supervisor')
while True:
    try:r=json.loads(supervisor.read_text(encoding='utf-8'))
    except json.JSONDecodeError:time.sleep(2);continue
    if r['status']=='completed':break
    if r['status']=='failed':status('training_failed',training=r);sys.exit(1)
    time.sleep(10)
status('verifying',supervisor=r)
with (O/'verifier_stdout_v1.log').open('xb') as stdout,(O/'verifier_stderr_v1.log').open('xb') as stderr:
    p=subprocess.Popen([sys.executable,'-B','-u',str(R/'scripts/verify_learning37.py')],cwd=str(R),stdout=stdout,stderr=stderr,
        creationflags=subprocess.CREATE_NO_WINDOW)
    status('verifying',verifier_launcher_pid=p.pid);code=p.wait()
status('completed' if code==0 else 'verification_failed',exit_code=code);sys.exit(code)
