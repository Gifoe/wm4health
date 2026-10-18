from common88_v1 import *
import ast,importlib.metadata
def main():
 manifest=read(O/'SOURCE_MANIFEST88_V1.json');source=Path(manifest['source_root']);classes=[];imports=set();configs=[]
 for row in manifest['files']:
  path=source/row['path'];assert sha(path)==row['sha256'],row['path']
  if path.suffix in ['.yaml','.yml']:
   if path.name in ['default_mts3.yaml','default_hiprssm.yaml','config.yaml','default.yaml']:configs.append(dict(path=row['path'],text=path.read_text(encoding='utf-8')))
  if path.suffix!='.py':continue
  try:tree=ast.parse(path.read_text(encoding='utf-8'))
  except SyntaxError:continue
  for node in ast.walk(tree):
   if isinstance(node,ast.Import):imports.update(n.name.split('.')[0] for n in node.names)
   elif isinstance(node,ast.ImportFrom) and node.module:imports.add(node.module.split('.')[0])
  if path.name not in ['MTS3.py','hipRSSM.py','acRKN.py']:continue
  for node in tree.body:
   if isinstance(node,ast.ClassDef):classes.append(dict(path=row['path'],name=node.name,line=node.lineno,methods=[dict(name=n.name,line=n.lineno,args=ast.unparse(n.args)) for n in node.body if isinstance(n,ast.FunctionDef)]))
 versions={}
 for name in ['torch','numpy','omegaconf','hydra-core','wandb','einops','PyYAML']:
  try:versions[name]=importlib.metadata.version(name)
  except importlib.metadata.PackageNotFoundError:versions[name]=None
 receipt=dict(utc=utc(),status='PINNED_NATIVE_SOURCE_AST_AUDITED_NOT_EXECUTED',source_commit=manifest['commit'],source_files_verified=len(manifest['files']),classes=classes,configuration_files=configs,import_roots=sorted(imports),available_runtime_versions=versions,models_executed=False,dataset_downloaded=False,source_manifest_sha256=sha(O/'SOURCE_MANIFEST88_V1.json'))
 write(O/'SOURCE_AST_AUDIT88_V1.json',receipt);log(dict(stage='source_ast_audit',source_commit=manifest['commit'],verified_files=len(manifest['files']),native_classes=classes,versions=versions,configs=[c['path'] for c in configs]))
 print(json.dumps(configs[:4],indent=2),flush=True)
if __name__=='__main__':main()
