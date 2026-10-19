"""Metadata/source-signature probe only; no numerical import or GPU context."""
import pathlib,sys,json,hashlib,datetime,importlib.util,importlib.metadata,ast
R=pathlib.Path(__file__).resolve().parents[1];O=R/'outputs';O.mkdir(exist_ok=True);out=O/'dependency_probe_v1.json';assert not out.exists();records=[]
for module,distribution in [('torch','torch'),('torchdiffeq','torchdiffeq'),('numpy','numpy'),('scipy','scipy'),('yaml','PyYAML'),('tqdm','tqdm'),('tensorboardX','tensorboardX')]:
 spec=importlib.util.find_spec(module)
 try:version=importlib.metadata.version(distribution)
 except importlib.metadata.PackageNotFoundError:version=None
 records.append({'module':module,'version':version,'origin':spec.origin if spec else None})
rk4=None;spec=importlib.util.find_spec('torchdiffeq')
if spec and spec.origin:
 source=pathlib.Path(spec.origin).parent/'_impl/rk_common.py'
 if source.is_file():
  text=source.read_text(encoding='utf-8');tree=ast.parse(text);matches=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='rk4_alt_step_func'];rk4={'present':len(matches)==1,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'signature':ast.unparse(matches[0].args) if matches else None}
result={'status':'METADATA_ONLY','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'python':sys.version,'dependencies':records,'private_RK4_source_symbol':rk4,'torch_imported':'torch' in sys.modules,'no_numerical_execution':True,'source_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()};assert not result['torch_imported'];out.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({'status':result['status'],'dependencies':[{k:r[k] for k in ['module','version']} for r in records],'private_RK4_symbol_present':rk4['present'] if rk4 else False,'torch_imported':False}))
