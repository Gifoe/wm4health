"""Run one experiment script with user-site and inherited DLL paths excluded."""
import sys,pathlib,site,runpy,ast,types,importlib.util
SCRIPTS=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(SCRIPTS));assert not site.ENABLE_USER_SITE;assert sys.flags.isolated
target=(SCRIPTS/pathlib.Path(sys.argv[1]).name).resolve();assert target.parent==SCRIPTS and target.is_file();sys.argv=[str(target),*sys.argv[2:]]
if not target.name.startswith('capture_'):
 # CSDI's configured full-attention path never invokes this optional package.
 # Omit only its unused import if absent; keep original source bytes and all
 # native definitions/algorithms. No fake dependency module is provided.
 if importlib.util.find_spec('linear_attention_transformer') is None:
  original=SCRIPTS.parent/'upstream'/'csdi'/'diff_models.py';tree=ast.parse(original.read_text());removed=[n for n in tree.body if isinstance(n,ast.ImportFrom) and n.module=='linear_attention_transformer'];assert len(removed)==1;tree.body=[n for n in tree.body if n not in removed];mod=types.ModuleType('diff_models');mod.__file__=str(original);sys.modules['diff_models']=mod;exec(compile(ast.fix_missing_locations(tree),str(original),'exec'),mod.__dict__)
runpy.run_path(str(target),run_name='__main__')
