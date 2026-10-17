import pathlib,sys,runpy,json,traceback,datetime,os
root=pathlib.Path(__file__).resolve().parents[1];out=root/'outputs'
sys.stdout=(out/'reference_control54_v1.stdout.log').open('x',encoding='utf-8',buffering=1);sys.stderr=(out/'reference_control54_v1.stderr.log').open('x',encoding='utf-8',buffering=1)
def state(status,**extra):(out/'reference_control54_v1.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':status,'pid':os.getpid(),**extra},indent=2),encoding='utf-8')
try:
 state('RUNNING_FROZEN_TRAINING_ONLY_REFERENCE_CONTROL');runpy.run_path(str(root/'scripts/audit_reference_control54_v1.py'),run_name='__main__');state('COMPLETE_NO_TRAINING_OR_HELDOUT_CHANGES')
except Exception:state('FAILED_PRESERVED',traceback=traceback.format_exc());raise
