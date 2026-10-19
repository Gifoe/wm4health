"""CPU-native compatibility inspection, no claim of a scientific reproduction."""
import os
os.environ['KERAS_BACKEND']='jax';os.environ['JAX_PLATFORMS']='cpu';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['WANDB_MODE']='disabled'
import datetime,importlib.metadata,json,pathlib,sys
import jax
jax.config.update('jax_enable_x64',True)
import keras
import numpy as np
import scipy
ROOT=pathlib.Path(__file__).resolve().parents[1]
report={'python':sys.version,'sys_prefix':sys.prefix,'backend':keras.backend.backend(),'jax_devices':[str(d) for d in jax.devices()],'jax_x64_enabled':bool(jax.config.x64_enabled),'packages':{p:importlib.metadata.version(p) for p in ['keras','jax','jaxlib','numpy','scipy','h5py']},'native_source':'sources/qg_closure','source_revision':'cf3d3e9272f8de536fd5de9ac2229589cb284fec','scientific_model_loaded':False,'scientific_inference_executed':False,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(ROOT/'outputs/native_runtime.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))
