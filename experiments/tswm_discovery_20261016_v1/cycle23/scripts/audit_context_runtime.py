import runtime_profile
import importlib,importlib.util,json,pathlib,sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
rows={}
for name in ['torch','torchvision','gym','torch_scatter','numpy','PIL']:
    try:
        if importlib.util.find_spec(name) is None:rows[name]={'available':False};continue
        mod=importlib.import_module(name);rows[name]={'available':True,'version':getattr(mod,'__version__',None),'path':getattr(mod,'__file__',None)}
    except Exception as e:rows[name]={'available':False,'error':repr(e)}
(ROOT/'outputs/native_runtime_audit.json').write_text(json.dumps(dict(python=sys.version,packages=rows),indent=2),encoding='utf-8');print(json.dumps(rows))
