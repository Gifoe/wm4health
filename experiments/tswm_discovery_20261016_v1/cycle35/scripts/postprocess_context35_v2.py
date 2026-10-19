"""Run independent audit and figure after the original V2 worker finishes."""
import pathlib,sys,time,json,subprocess,os,datetime
ROOT=pathlib.Path(__file__).resolve().parents[1];out=ROOT/'outputs';path=out/'postprocess_v2_status.json'
assert not path.exists(),'Duplicate postprocess refused'
def status(value):
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'pid':os.getpid(),**value},indent=2),encoding='utf-8');temp.replace(path)
status({'status':'waiting_for_original_training'})
while True:
    p=out/'supervisor_v2.json'
    if p.exists():
        try:rec=json.loads(p.read_text())
        except json.JSONDecodeError:time.sleep(2);continue
        if rec['status']=='failed':status({'status':'training_failed','training_record':rec});sys.exit(1)
        if rec['status']=='completed':break
    time.sleep(2)
for script in ['verify_context35_v2.py','plot_context35_v2.py']:
    status({'status':'running','script':script})
    p=subprocess.run([sys.executable,'-B','-u',str(ROOT/'scripts'/script)],cwd=str(ROOT),capture_output=True)
    (out/(script+'.stdout.log')).write_bytes(p.stdout);(out/(script+'.stderr.log')).write_bytes(p.stderr)
    if p.returncode:
        status({'status':'failed','script':script,'exit_code':p.returncode});sys.exit(p.returncode)
status({'status':'completed'})
