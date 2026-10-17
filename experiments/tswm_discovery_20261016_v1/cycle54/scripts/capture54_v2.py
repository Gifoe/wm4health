"""Persistent remote WMI entrypoint with in-process log capture."""
import pathlib,sys,runpy,json,datetime
root=pathlib.Path(__file__).resolve().parents[1];out=root/'outputs';out.mkdir(exist_ok=True)
sys.stdout=(out/'supervisor54_v2.stdout.log').open('x',encoding='utf-8',buffering=1)
sys.stderr=(out/'supervisor54_v2.stderr.log').open('x',encoding='utf-8',buffering=1)
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'PERSISTENT_WMI_CAPTURE_ENTERED','initial_start_process_attempt':'No surviving process, empty logs, no preparation or fit artifacts observed. Initial logs retained.'}),flush=True)
runpy.run_path(str(root/'scripts/supervise54.py'),run_name='__main__')
