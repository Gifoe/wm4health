import pathlib,subprocess,sys,json,datetime,hashlib
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs';OUT.mkdir(exist_ok=True,parents=True)
with (OUT/'stage_model66_v1_stdout.log').open('xb') as f:
 child=subprocess.Popen([sys.executable,'-B','-u',str(ROOT/'scripts'/'stage_model66_v1.py')],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT)
 (OUT/'stage_launch66_v1.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':child.pid,'status':'CHILD_LIVE'},indent=2));code=child.wait()
with (OUT/'stage_exit66_v1.json').open('x') as f:json.dump({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':child.pid,'exit_code':code},f,indent=2)
