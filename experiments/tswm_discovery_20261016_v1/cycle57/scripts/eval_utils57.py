from common57 import *
import numpy as np
import torch
from models57 import Neural,Proper
from math57 import posterior,rho_from
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False

def load_model(record):
    assert sha(record['checkpoint'])==record['checkpoint_sha256']
    payload=torch.load(record['checkpoint'],map_location='cpu',weights_only=False)
    assert payload['sensor']==record['sensor'] and payload['method']==record['method']
    dtype=torch.float64 if record['method']=='proper' else torch.float32
    model=(Proper() if record['method']=='proper' else Neural()).to(device='cuda',dtype=dtype)
    model.load_state_dict(payload['model'],strict=True);model.eval()
    return model,dtype

@torch.no_grad()
def infer(model,dtype,x,actions):
    chunks=[]
    for first in range(0,len(x),128):
        xx=torch.tensor(x[first:first+128],device='cuda',dtype=dtype)
        act=torch.tensor(actions[first:first+128],device='cuda',dtype=torch.long)
        result=posterior(model.theta,model.lam,model.encode(xx),act)
        chunks.append(result['m'].cpu().numpy().astype('float64'))
    return np.concatenate(chunks)

def parameter_values(model):
    return float(rho_from(model.theta[0]).detach()),model.theta[1:3].detach().cpu().numpy().astype('float64')

def roll(mean,rho,angles,actions):
    N,H=actions.shape;out=np.empty((N,H+1,2));out[:,0]=mean
    for h in range(H):
        angle=angles[actions[:,h]];c=np.cos(angle);s=np.sin(angle)
        matrices=rho*np.stack([c,-s,s,c],-1).reshape(N,2,2)
        out[:,h+1]=np.einsum('nij,nj->ni',matrices,out[:,h])
    return out

def qualify_roll():
    m=np.array([[.3,-.7],[.1,.2]]);rho=.8;angles=np.array([.2,-.4]);errors=[]
    for action in [0,1]:
        theta=angles[action];A=rho*np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        trajectory=roll(m,rho,angles,np.full((2,20),action,dtype=int))
        for h in [1,5,10,20]:errors.append(np.max(abs(trajectory[:,h]-m@np.linalg.matrix_power(A,h).T)))
    assert max(errors)<1e-14
    return float(max(errors))
