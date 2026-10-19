"""Offline task-local target installation. No preexisting environment changes."""
import hashlib,json,pathlib,subprocess,sys
ROOT=pathlib.Path(__file__).resolve().parents[1];WH=ROOT/'dependency_wheels';target=ROOT/'vendor/native_control'
assert not target.exists(),'Refuse modifying an existing dependency target'
manifest=json.loads((WH/'manifest.json').read_text())
for row in manifest:
 assert hashlib.sha256((WH/row['filename']).read_bytes()).hexdigest()==row['sha256']
command=[sys.executable,'-m','pip','install','--no-index','--no-deps','--no-compile','--find-links',str(WH),'--target',str(target),'numba==0.61.2','llvmlite==0.44.0']
completed=subprocess.run(command,check=True,capture_output=True,text=True)
result=dict(status='completed',command=command,stdout=completed.stdout,stderr=completed.stderr,wheels=manifest,
            scope='Only new cycle26/vendor/native_control directory modified; main environment unchanged')
(ROOT/'outputs/native_control_dependency_install.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(dict(status=result['status'],stdout=completed.stdout,scope=result['scope'])),flush=True)
