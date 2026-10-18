"""Dependency-only reroute and a new artifact version for the preserved qualification."""
import sys,pathlib,ast
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common72_v2 import *
source=ROOT/'scripts'/'qualify_native_contract72_v1.py';tree=ast.parse(source.read_bytes());changes=[]
for n in ast.walk(tree):
    if isinstance(n,ast.ImportFrom) and n.module=='common72_v1':n.module='common72_v2';changes.append('project-local dependency prefix')
    elif isinstance(n,ast.Constant) and n.value=='native_contract_v1':n.value='native_contract_v2';changes.append('new synthetic artifact directory')
    elif isinstance(n,ast.Constant) and n.value=='NATIVE_CONTRACT_QUALIFICATION72_V1.json':n.value='NATIVE_CONTRACT_QUALIFICATION72_V2.json';changes.append('new output version')
assert len(changes)==3
dump(OUT/'NATIVE_QUALIFICATION_ADAPTER72_V2.json',dict(utc=utc(),original_qualification_sha256=sha(source),changes=changes,
    original_failure='ModuleNotFoundError: pandas before executing any native benchmark function',
    dependency_receipt_sha256=sha(OUT/'DEPENDENCIES72_V1.json'),benchmark_algorithm_bodies_changed=False))
exec(compile(ast.fix_missing_locations(tree),str(source)+'::scoped_dependencies_v2','exec'),dict(__name__='__main__',__file__=str(source)))
