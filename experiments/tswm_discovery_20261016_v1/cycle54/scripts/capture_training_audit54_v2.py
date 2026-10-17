import pathlib,sys,runpy,json,traceback,datetime,os
root=pathlib.Path(__file__).resolve().parents[1];out=root/'outputs'
sys.stdout=(out/'training_audit54_v2.stdout.log').open('x',encoding='utf-8',buffering=1);sys.stderr=(out/'training_audit54_v2.stderr.log').open('x',encoding='utf-8',buffering=1)
def state(status,**extra):(out/'training_audit54_v2.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':status,'pid':os.getpid(),**extra},indent=2),encoding='utf-8')
try:
 state('RUNNING_FROZEN_TRAINING_ONLY_PROBE');runpy.run_path(str(root/'scripts/audit_training_objective54_v2.py'),run_name='__main__');state('COMPLETE_NO_TRAINING_OR_HELDOUT_CHANGES')
except Exception:state('FAILED_PRESERVED',traceback=traceback.format_exc());raise
