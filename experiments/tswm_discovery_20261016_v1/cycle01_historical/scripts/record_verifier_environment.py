import importlib.metadata,json,pathlib,platform
ROOT=pathlib.Path(__file__).resolve().parents[1]
packages={name:importlib.metadata.version(name) for name in ['torch','numpy','scipy','packaging','tqdm','graphviz','appdirs']}
result={'python':platform.python_version(),'packages':packages,'device':'cpu','upstream_revision':'e3930353a3589f7d10879f1e71223d1eefb57196','isolated_environment':'.venv_verifier'}
(ROOT/'outputs/verifier_environment.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
(ROOT/'requirements_verifier.txt').write_text('--extra-index-url https://download.pytorch.org/whl/cpu\n'+'\n'.join(f'{k}=={v}' for k,v in packages.items())+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))
