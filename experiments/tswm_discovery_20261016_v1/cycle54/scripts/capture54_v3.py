"""Persistent entrypoint after verifying and completing pinned-source staging."""
import pathlib,sys,runpy,json,datetime,hashlib
root=pathlib.Path(__file__).resolve().parents[1];out=root/'outputs'
sys.stdout=(out/'supervisor54_v3.stdout.log').open('x',encoding='utf-8',buffering=1)
sys.stderr=(out/'supervisor54_v3.stderr.log').open('x',encoding='utf-8',buffering=1)
old=out/'state54.json';archive=out/'attempt54_v2_state.json';assert old.exists() and not archive.exists();archive.write_bytes(old.read_bytes())
assert (out/'preflight54.json').exists()
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'RESTART_AFTER_SOURCE_PREFLIGHT_PASS','prior_attempt':'Preparation failed because the remote source staging omitted the two official demo notebooks. Filled only those missing public files from the pinned archive; all 21 hashes verified. No previous source replaced; no model fitting had started.','prior_attempt_state_sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}),flush=True)
runpy.run_path(str(root/'scripts/supervise54.py'),run_name='__main__')
