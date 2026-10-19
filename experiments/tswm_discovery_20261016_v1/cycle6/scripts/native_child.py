"""Execute unchanged native mathematical class bodies; no external logger launcher.

Lightning's container base is adapted to nn.Module for manual Torch orchestration.
No neural layer, loss, reparameterization, Jacobian or optimizer body is replaced.
This is disclosed source-faithful model execution, not exact launcher replication.
"""
import runtime_profile
import ast,pathlib,types
import torch,numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.distributions as D
from torch.nn import functional as F
from torch.autograd import Variable
from torch.func import vmap,jacfwd
ROOT=pathlib.Path(__file__).resolve().parents[1]
SOURCE=ROOT/'sources/child/synthetic/CHiLD'
def select(path,names,namespace):
 tree=ast.parse(path.read_text(encoding='utf-8'))
 nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
 assert set(n.name for n in nodes)==set(names),(path,names)
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
 return namespace
def namespace():return dict(torch=torch,np=np,nn=nn,init=init,D=D,F=F,Variable=Variable,vmap=vmap,jacfwd=jacfwd,pl=types.SimpleNamespace(LightningModule=nn.Module))
def classes():
 env=namespace()
 select(SOURCE/'modules/components/mlp.py',['NLayerLeakyMLP'],env)
 select(SOURCE/'modules/components/beta.py',['reparametrize','kaiming_init','BetaVAE_MLP'],env)
 select(SOURCE/'modules/components/transition.py',['NPInstantaneousTransitionPrior','NPChangeInstantaneousTransitionPrior'],env)
 select(SOURCE/'modules/instantaneous.py',['InstantaneousProcess'],env)
 return env
def build(seed=None,device='cuda'):
 if seed is not None:torch.manual_seed(seed)
 cls=classes()['InstantaneousProcess']
 model=cls(input_dim=4,z_dim=7,z_dim_fix=7,z_dim_change=0,lag=1,nclass=15,hidden_dim=128,embedding_dim=2,lr=3e-4,beta=.008,gamma=.008,theta=.0023,decoder_dist='gaussian',correlation='Spearman',n_mea=5,z_dim_list=[1,2,4],optimizer_type='adam')
 return model.to(device)
