"""Install only the five audited wheels into this cycle's private vendor folder."""
import pathlib,json,hashlib,subprocess,sys
R=pathlib.Path(__file__).resolve().parents[1];record=json.loads((R/'assets/WHEEL_RECEIPTS.json').read_text());files=[]
for x in record['files']:
 p=R/'assets'/x['file'];assert hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256'];files.append(str(p))
assert not (R/'vendor').exists()
subprocess.run([sys.executable,'-m','pip','install','--no-deps','--no-index','--target',str(R/'vendor'),*files],check=True)
print('Project-local wheels installed; global runtime unchanged',flush=True)
