"""Execute unchanged author class/function bodies without unused framework imports.

Only imports/package initializers for Lightning, dataset frameworks and unused
Reformer attention are bypassed. Native Forecaster, attention, scaler, model,
loss and sampler bodies are retained. This is not a replacement architecture.
"""
import ast,math,pathlib,typing
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal,MultivariateNormal
from einops import rearrange,repeat
ROOT=pathlib.Path(__file__).resolve().parents[1]
SRC=ROOT/'sources'/'cofe'/'probts'
def select(path,names,namespace):
 tree=ast.parse(path.read_text(encoding='utf-8'))
 nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
 assert {n.name for n in nodes}==set(names),(path,names)
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
def load():
 ns={'torch':torch,'nn':nn,'F':F,'np':np,'math':math,'sqrt':math.sqrt,'Normal':Normal,'MultivariateNormal':MultivariateNormal,'rearrange':rearrange,'repeat':repeat,'List':typing.List,'Union':typing.Union,'Optional':typing.Optional}
 select(SRC/'utils'/'utils.py',['weighted_average'],ns)
 select(SRC/'data'/'data_utils'/'data_scaler.py',['Scaler','TemporalScaler'],ns)
 select(SRC/'model'/'forecaster'/'forecaster.py',['Forecaster'],ns)
 sub=SRC/'model'/'nn'/'prob'/'k2VAE'
 select(sub/'masking.py',['TriangularCausalMask','ProbMask'],ns)
 select(sub/'SelfAttention_Family.py',['FullAttention','AttentionLayer'],ns)
 select(sub/'Transformer_EncDec.py',['Encoder','EncoderLayer'],ns)
 select(sub/'auxiliarynet.py',['Transformer'],ns);ns['AuxiliaryNet']=ns['Transformer']
 select(SRC/'model'/'nn'/'prob'/'RevIN.py',['RevIN'],ns)
 select(SRC/'model'/'nn'/'prob'/'COFE.py',['latent_transitionerNN','transition_zNN','MLP','MLP2','estimatenoise','COFEVAE'],ns)
 select(SRC/'model'/'forecaster'/'prob_forecaster'/'COFE.py',['ConvertedParams','COFEModel'],ns)
 return ns
def build(device='cuda',seed=41):
 torch.manual_seed(seed);ns=load()
 return ns['COFEModel'](d_model=128,d_ff=256,e_layers=1,dropout=.1,activation='relu',n_heads=8,factor=3,patch_len=8,multistep=True,dynamic_dim=128,hidden_layers=2,hidden_dim=256,weight_beta=.01,weight_l1=.01,weight_con=.1,zero_threshold=1e-4,sample_schedule=10,target_dim=7,context_length=24,prediction_length=24,freq='h',lags_list=[]).to(device)
