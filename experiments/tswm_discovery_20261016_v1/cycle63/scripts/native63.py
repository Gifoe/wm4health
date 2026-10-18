from common63 import *
import importlib,sys,types,ast,copy
import numpy as np
import torch
import torch.nn.functional as F
UP=ROOT/'upstream/cofe'
def load():
 sources()
 for name in ['probts','probts.model','probts.model.nn','probts.model.nn.prob','probts.model.nn.prob.k2VAE']:
  m=types.ModuleType(name);m.__path__=[str(UP.joinpath(*name.split('.')))];m.__package__=name;sys.modules[name]=m
 Core=importlib.import_module('probts.model.nn.prob.COFE').COFEVAE
 # Execute the exact selected method ASTs, without stubbing the unavailable framework.
 p=UP/'probts/model/forecaster/prob_forecaster/COFE.py';tree=ast.parse(p.read_text(encoding='utf-8'))
 cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='COFEModel')
 methods=[copy.deepcopy(n) for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ['forward','kld_loss','loss']]
 ns={'torch':torch,'F':F};exec(compile(ast.Module(body=methods,type_ignores=[]),str(p),'exec'),ns)
 class Harness(torch.nn.Module):
  def __init__(self,core):
   super().__init__();self.model=core;self.context_length=96;self.weight_beta=.01;self.weight_l1=.001;self.weight_con=.01
 for n in ['forward','kld_loss','loss']:setattr(Harness,n,ns[n])
 return Core,Harness,{'source_method_file':str(p),'selected_method_ast_names':[n.name for n in methods],'full_forecaster_or_Lightning_imported':False,'native_core_imported_unchanged':True}
def config():
 return types.SimpleNamespace(seq_len=96,pred_len=720,patch_len=24,multistep=True,dynamic_dim=128,hidden_layers=3,hidden_dim=256,n_vars=7,d_model=128,d_ff=256,e_layers=1,dropout=.1,activation='relu',n_heads=4,factor=3,sample_schedule=20,zero_threshold=1e-5)
def seed(s):np.random.seed(s);torch.manual_seed(s);torch.set_num_threads(4)
def fixture():
 import pandas as pd
 p=PROJECT/'cycle5/data/ETTh1.csv';b=p.read_bytes();assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()=='a52c4925778c07c1ef1a2cf6fd01594919717d9e'
 raw=pd.read_csv(p).drop(columns=['date']).values;mu=raw[:2048].mean(0);sd=raw[:2048].std(0);z=(raw-mu)/sd;origins=[96,128,256,384]
 x=torch.tensor(np.stack([z[o-96:o] for o in origins]),dtype=torch.float64);y=torch.tensor(np.stack([z[o:o+720] for o in origins]),dtype=torch.float64)
 return x,y,{'source_path':str(p),'sha256':sha(p),'normalization_rows':[0,2048],'origins':origins,'max_used_row':1104,'shape_context':list(x.shape),'shape_target':list(y.shape),'scope':'early training-region fixture, no heldout data or training performed'}
