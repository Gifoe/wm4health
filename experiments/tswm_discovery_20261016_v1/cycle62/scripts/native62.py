from common62 import *
import importlib,sys,types,random
import numpy as np
import torch
UP=PROJECT/'upstream/PRO-DYN'
def load_native():
 compat=[]
 if not hasattr(np,'Inf'):np.Inf=float('inf');compat.append('numpy.Inf runtime constant alias; upstream bytes unchanged')
 for name in ['ts_benchmark','ts_benchmark.baselines','ts_benchmark.baselines.time_series_library','ts_benchmark.baselines.time_series_library.models','ts_benchmark.baselines.time_series_library.layers','ts_benchmark.baselines.time_series_library.utils','ts_benchmark.models','ts_benchmark.utils']:
  if name not in sys.modules:
   m=types.ModuleType(name);m.__path__=[str(UP.joinpath(*name.split('.')))];m.__package__=name;sys.modules[name]=m
 ad=importlib.import_module('ts_benchmark.baselines.time_series_library.adapters_for_transformers')
 models={a:getattr(importlib.import_module('ts_benchmark.baselines.time_series_library.models.'+a),a) for a in ['Informer','Informer_PRO','Informer_DYN']}
 return ad,models,compat
def seed(s):
 random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
 torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=False
def load_data():
 import pandas as pd
 p=PROJECT/'cycle5/data/ETTh1.csv';pin=read(ROOT/'PINNED_SOURCES62_V1.json')['ett'];b=p.read_bytes()
 assert len(b)==pin['size'] and blob(b)==pin['git_blob'], 'ETTh1 official data mismatch'
 df=pd.read_csv(p,parse_dates=['date']).set_index('date');assert df.shape==(17420,7) and np.isfinite(df.values).all();assert df.index.is_monotonic_increasing and df.index.is_unique
 assert (np.diff(df.index.values)==np.timedelta64(1,'h')).all()
 return df,p
def adapter(ad,models,arm):
 return ad.TransformerAdapter(arm,models[arm],seq_len=96,horizon=96,d_model=32,d_ff=128,norm=True,parallel_strategy=None)
def verify_sources():
 pin=read(ROOT/'PINNED_SOURCES62_V1.json');rows=[]
 for r in pin['files']:
  p=UP/r['path'];b=p.read_bytes();assert len(b)==r['size'] and blob(b)==r['git_blob'],r['path'];rows.append({**r,'sha256':hashlib.sha256(b).hexdigest()})
 return rows
def imported_sources():
 rows=[]
 for n,m in sorted(sys.modules.items()):
  f=getattr(m,'__file__',None)
  if f and pathlib.Path(f).resolve().is_relative_to(UP.resolve()):
   p=pathlib.Path(f);rows.append({'module':n,'path':p.relative_to(UP).as_posix(),'sha256':sha(p)})
 return rows
