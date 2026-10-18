from common89_v1 import *
import copy, importlib.util, types
import numpy as np
import torch

torch.set_num_threads(2)
torch.set_float32_matmul_precision('highest')
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
PRIOR = np.array([.4,.2,.1,.08,.07,.05,.04,.035,.025], np.float64)
KS = np.linspace(.4,2.,9)
LENGTH = 72
CONTEXT = 8

def source_setup():
    manifest = read(O/'CSDI_SOURCE89_V1.json')
    root = Path(manifest['source_root'])
    hashes = {p:sha(root/p) for p in ['main_model.py','diff_models.py']}
    shim = False
    if importlib.util.find_spec('linear_attention_transformer') is None:
        stub = types.ModuleType('linear_attention_transformer')
        class ForbiddenLinearAttention(torch.nn.Module):
            def __init__(self,*args,**kwargs):
                raise RuntimeError('Unused optional linear attention path must not be invoked')
        stub.LinearAttentionTransformer = ForbiddenLinearAttention
        sys.modules['linear_attention_transformer'] = stub
        shim = True
    sys.path.insert(0,str(root))
    from main_model import CSDI_base
    return CSDI_base,dict(root=str(root),revision='7f24a436f08d98853a6b43d4f7f04e5a65ecdf27',sha256=hashes,unused_optional_import_sentinel=shim)

def config():
    return dict(model=dict(timeemb=128,featureemb=16,is_unconditional=0,target_strategy='test'),diffusion=dict(layers=4,channels=64,nheads=8,diffusion_embedding_dim=128,beta_start=.0001,beta_end=.5,num_steps=50,schedule='quad',is_linear=False))

def new_model(seed,device='cuda'):
    torch.manual_seed(seed)
    cls,meta = source_setup()
    model = cls(4,copy.deepcopy(config()),device).to(device)
    return model,meta

def mask_and_time(n,device):
    mask = torch.ones((n,4,LENGTH),device=device)
    mask[:,:2,CONTEXT:] = 0
    tp = torch.arange(LENGTH,device=device,dtype=torch.float32).expand(n,-1)
    return mask,tp

def simulate(actions,world,family):
    folder = ROOT/'cycle88/scripts'
    if str(folder) not in sys.path:sys.path.insert(0,str(folder))
    from plan_data88_v1 import advance
    n = len(actions)
    state = np.zeros((n,3),np.float64)
    theta = np.tile(np.array([1.,.35,1.,.5,3.]),(n,1))
    theta[:,0] = KS[np.broadcast_to(np.asarray(world),(n,))]
    out = np.zeros((n,LENGTH,2),np.float64)
    a = np.zeros((n,2),np.float64)
    for t in range(CONTEXT,LENGTH):
        a[:,0] = actions[:,t]
        state = advance(state,a,theta,family)
        out[:,t] = state[:,:2]
    return out

def draw_actions(rng,n):
    actions = np.zeros((n,LENGTH),np.float64)
    actions[:,CONTEXT:] = np.repeat(rng.uniform(-1,1,(n,8)),8,axis=1)
    return actions

def features(states,actions,family,center,scale):
    result = np.zeros((len(states),4,LENGTH),np.float32)
    result[:,:2] = ((states-center)/scale).transpose(0,2,1)
    result[:,2] = actions
    result[:,3] = -1. if family==0 else 1.
    return result

@torch.no_grad()
def sample(model,observed,seed,batch=64):
    model.eval()
    outs=[]
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(seed)
        for start in range(0,len(observed),batch):
            x = torch.as_tensor(observed[start:start+batch],device=model.device)
            cm,tp = mask_and_time(len(x),model.device)
            side = model.get_side_info(tp,cm)
            y = model.impute(x,cm,side,1)[:,0,:2,CONTEXT:]
            assert torch.isfinite(y).all()
            outs.append(y.cpu().numpy())
    return np.concatenate(outs)
