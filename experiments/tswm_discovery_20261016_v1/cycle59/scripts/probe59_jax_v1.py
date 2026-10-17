"""Read installed metadata without installing or changing the existing JAX runtime."""
from common59 import *
import importlib.util,importlib.metadata,sys
packages={}
for name in ['numpy','scipy','pandas','jax','jaxlib','equinox','diffrax','optimistix','optax']:
    try:packages[name]={'present':bool(importlib.util.find_spec(name)),'version':importlib.metadata.version(name)}
    except importlib.metadata.PackageNotFoundError:packages[name]={'present':False,'version':None}
obj={'utc':utc(),'status':'READ_ONLY_JAX_RUNTIME_METADATA_PROBE','python':sys.version,'executable':sys.executable,'packages':packages,'environment_changes':False}
dump(OUT/'JAX_RUNTIME59_V1.json',obj)
print(json.dumps(obj),flush=True)
