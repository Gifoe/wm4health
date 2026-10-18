"""Verify pinned selected public source; never vendor it into published artifacts."""
import sys,pathlib,ast
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common72_v1 import *
selection=read(ROOT/'SOURCE_SELECTION72_V1.json');upstream=PROJECT/'upstream'/'epi-cf-benchmark'
original_pin=read(upstream/'SOURCE_PIN.json');assert original_pin['revision']==selection['revision']
rows=[]
for spec in selection['files']:
    p=upstream/spec['path'];b=p.read_bytes();assert len(b)==spec['bytes'] and hashlib.sha256(b).hexdigest()==spec['sha256']
    assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==spec['git_blob_sha1']
    definitions={}
    if p.suffix=='.py':
        tree=ast.parse(b)
        definitions={n.name:hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest() for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    rows.append(dict(**spec,definitions=definitions))
result=dict(utc=utc(),status='PINNED_NATIVE_EPICF_SOURCE_VERIFIED',repository=selection['repository'],revision=selection['revision'],tree=selection['tree'],
    files=rows,selected_files=len(rows),selected_bytes=sum(r['bytes'] for r in rows),source_pin_sha256=sha(upstream/'SOURCE_PIN.json'),
    no_license_file_in_inspected_tree=True,original_source_redistributed=False,source_imports_executed=False,
    numeric_target_data_accessed=False,local_result_data_or_model_files_written=0)
dump(OUT/'SOURCE_AUDIT72_V1.json',result)
print(json.dumps({k:result[k] for k in ['status','revision','selected_files','selected_bytes','original_source_redistributed']}),flush=True)
