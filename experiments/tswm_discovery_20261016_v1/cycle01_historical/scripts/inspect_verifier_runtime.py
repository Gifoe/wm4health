import importlib.metadata,json,platform
result={'python':platform.python_version()}
for name in ['torch','numpy','appdirs','graphviz','tqdm']:
    try:
        result[name]=importlib.metadata.version(name)
    except Exception as e:result[name]=type(e).__name__
print(json.dumps(result))
