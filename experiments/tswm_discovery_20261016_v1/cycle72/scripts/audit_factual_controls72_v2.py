"""Preserved factual protocol with explicit CPU allocation and a versioned receipt."""
import sys,pathlib,ast
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common72_v3 import *
source=ROOT/'scripts'/'audit_factual_controls72_v1.py';tree=ast.parse(source.read_bytes());changes=[];body=[]
for n in tree.body:
    if isinstance(n,ast.ImportFrom) and n.module=='common72_v2':n.module='common72_v3';changes.append('CPU-only environment prefix')
    if isinstance(n,ast.Assert) and ast.unparse(n.test)=='not torch.cuda.is_available()':
        body.extend(ast.parse("torch.set_default_device('cpu')\nassert torch.get_default_device().type == 'cpu'").body)
        changes.append('Validate actual tensor allocation default, not only device visibility')
    else:body.append(n)
tree.body=body
for n in ast.walk(tree):
    if isinstance(n,ast.Constant) and n.value=='72_V1.json':n.value='72_V2.json'
    elif isinstance(n,ast.Constant) and n.value=='FACTUAL_CONTROLS72_V1.json':n.value='FACTUAL_CONTROLS72_V2.json'
    elif isinstance(n,ast.keyword) and n.arg=='cuda_visible':n.arg='cuda_available_observed';n.value=ast.parse('bool(torch.cuda.is_available())',mode='eval').body
# Add a direct native data-device check before fitting, without changing its body.
for node in ast.walk(tree):
    if isinstance(node,ast.For) and ast.unparse(node.target).startswith('(setting,'):
        revised=[]
        for n in node.body:
            if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='log' for t in n.targets):
                revised.extend(ast.parse("native_cpu_data = kde._dataset(native_data, args, 'train')\nassert all(x.device.type == 'cpu' for x in native_cpu_data.inputs + native_cpu_data.targets)\ndel native_cpu_data").body)
                changes.append('Assert every native training tensor is on CPU')
            revised.append(n)
        node.body=revised
assert len(changes)==3,changes
dump(OUT/'FACTUAL_CONTROLS_ADAPTER72_V2.json',dict(utc=utc(),source_adapter_sha256=sha(source),changes=changes,
    diagnostic_features_and_split_changed=False,native_algorithm_bodies_changed=False,
    first_failure='A device-visibility assertion fired before reading numeric factual data; no GPU tensor computation had executed'))
exec(compile(ast.fix_missing_locations(tree),str(source)+'::cpu_allocation_v2','exec'),dict(__name__='__main__',__file__=str(source)))
