from common64 import *
import ast,sys,types,importlib
import numpy as np
import torch
def load():
 source_check();up=ROOT/'upstream/tds/smc_utils';pkg=types.ModuleType('smc_utils');pkg.__path__=[str(up)];sys.modules['smc_utils']=pkg
 p=up/'smc_utils.py';tree=ast.parse(p.read_text(encoding='utf-8'));removed=[];body=[]
 for n in tree.body:
  if isinstance(n,ast.ImportFrom) and n.module=='numba':removed.append('optional numba import');continue
  if isinstance(n,ast.FunctionDef) and n.name=='inverse_cdf':
   assert len(n.decorator_list)==1;n.decorator_list=[];removed.append('optional jit decorator on inverse_cdf; Python algorithm unchanged')
  body.append(n)
 tree.body=body;m=types.ModuleType('smc_utils.smc_utils');m.__file__=str(p);m.__package__='smc_utils';sys.modules[m.__name__]=m;exec(compile(ast.fix_missing_locations(tree),str(p),'exec'),m.__dict__)
 fk=importlib.import_module('smc_utils.feynman_kac_pf').smc_FK
 return fk,m,removed
