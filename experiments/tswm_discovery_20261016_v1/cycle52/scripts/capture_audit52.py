"""Detached audit supervisor with durable logs and terminal small-report mirror."""
import datetime
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys

R=pathlib.Path(__file__).resolve().parents[1]
state=R/'outputs/audit_supervisor_v1.json'
assert not state.exists(), 'Inspect existing supervisor instead of restarting'
def save(status,**kwargs):
    state.write_text(json.dumps({'status':status,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),**kwargs},indent=2),encoding='utf-8')
env=os.environ.copy();env['PYTHONPATH']=str(R/'runtime_deps')+os.pathsep+env.get('PYTHONPATH','')
save('STARTING')
with (R/'outputs/native_audit_v3.log').open('wb') as log:
    child=subprocess.Popen([sys.executable,'-B','-u',str(R/'scripts/audit_greenhouse52_v3.py')],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT)
    save('RUNNING',child_pid=child.pid,log='native_audit_v3.log')
    code=child.wait()
if code:
    save('FAILED',child_pid=child.pid,exit_code=code)
    sys.exit(1)
storage=json.loads((R.parent/'STORAGE_LOCATION_V1.json').read_text())
out=pathlib.Path(storage['artifact_root'])/'cycle52/native_audit_v3'
audit=json.loads((out/'audit.json').read_text())
assert audit['status']=='COMPLETE_REAL_DATA_INPUT_CONTRACT_AUDIT'
mirror=R/'outputs/native_audit_complete_v3';assert not mirror.exists();mirror.mkdir()
manifest=[]
for source in out.iterdir():
    if source.is_file() and source.suffix in ['.json','.csv','.md','.txt']:
        target=mirror/source.name;shutil.copyfile(source,target)
        assert source.read_bytes()==target.read_bytes()
        manifest.append({'path':target.name,'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
(mirror/'mirror_manifest.json').write_text(json.dumps({'large_disk_source':str(out),'files':manifest},indent=2),encoding='utf-8')
save('COMPLETE_REAL_DATA_INPUT_CONTRACT_AUDIT',child_pid=child.pid,exit_code=code,large_disk_output=str(out),report_mirror=str(mirror),overall=audit['overall'])
