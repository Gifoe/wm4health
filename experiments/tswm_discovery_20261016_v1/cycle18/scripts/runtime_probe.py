"""Read-only interpreter and import availability inventory."""
import importlib.util,json,sys
print(json.dumps({'version':sys.version.split()[0],
    'modules':{name:importlib.util.find_spec(name) is not None for name in
        ['torch','safetensors','numpy','scipy','sklearn','statsmodels','huggingface_hub']}}))
