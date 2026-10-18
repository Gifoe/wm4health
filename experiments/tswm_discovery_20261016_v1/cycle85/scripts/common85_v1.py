import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES','-1')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import ast,sys,types,json,hashlib,datetime,copy,itertools
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
torch.set_num_threads(2)
ROOT=Path(__file__).resolve().parents[2];C=ROOT/'cycle85';O=C/'outputs'
CONFIG=json.loads((ROOT/'STORAGE_LOCATION_V1.json').read_text(encoding='utf-8-sig'))
RAW=Path(CONFIG['artifact_root'])/'cycle85'/'anchored_screen_v1'
DATA=Path(CONFIG['data_root'])/'cycle85'/'anchored_screen_v1'
WEIGHTS=Path(CONFIG['checkpoint_root'])/'cycle85'/'anchored_screen_v1'
DT=.05
SEEDS=[2026851,2026852]
SHIFT_CASES=['none','observation','dynamics','joint']
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,obj,replace=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    b=json.dumps(obj,indent=2,allow_nan=False).encode()
    if replace:
        tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_bytes(b);tmp.replace(path)
    else:
        with path.open('xb') as f:f.write(b)
def log(obj):
    O.mkdir(parents=True,exist_ok=True);obj=dict(utc=utc(),**obj)
    with (O/'trace85_v1.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(obj,allow_nan=False)+'\n')
    print(json.dumps(obj,allow_nan=False),flush=True)
def native_modules():
    m=json.loads((ROOT/'cycle84'/'outputs'/'SOURCE_CONTRACT84_V1.json').read_text())
    r=next(z for z in m['repositories'] if z['repository']=='ExistentialRobotics/cross_embodiment_transfer')
    assert r['commit']=='3988511598a39c8a8a6530055b72609bf1b8b373'
    files={z['path']:z for z in r['selected']}
    for n in ['utils.py','td3.py','bc.py','align.py']:assert sha(files[n]['server_cache'])==files[n]['sha256']
    p=Path(files['utils.py']['server_cache']);tree=ast.parse(p.read_text())
    names={'Activation','_str_to_activation','AttrDict','soft_update_params','weight_init','build_mlp'}
    nodes=[]
    for n in tree.body:
        if isinstance(n,(ast.Import,ast.ImportFrom)):
            module=n.module if isinstance(n,ast.ImportFrom) else ' '.join(z.name for z in n.names)
            if 'robosuite' not in module:nodes.append(n)
        elif isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names:nodes.append(n)
        elif isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in n.targets):nodes.append(n)
    u=types.ModuleType('utils');u.__file__=str(p);sys.modules['utils']=u
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(p),'exec'),u.__dict__)
    answer=[]
    for name in ['td3','bc','align']:
        p=Path(files[name+'.py']['server_cache']);mod=types.ModuleType(name);mod.__file__=str(p);sys.modules[name]=mod
        exec(compile(p.read_text(),str(p),'exec'),mod.__dict__);answer.append(mod)
    return answer[1],answer[2],{n:files[n]['sha256'] for n in ['utils.py','td3.py','bc.py','align.py']}
def field(x,u,family,changed):
    if family=='duffing':
        k,c=(1.5,.2) if changed else (1.,.4)
        return np.stack([x[...,1],-k*x[...,0]-c*x[...,1]-.2*x[...,0]**3+.7*u[...,0]],axis=-1)
    b=2.4 if changed else 1.5;a=x[...,0];v=x[...,1]
    return np.stack([1-(b+1)*a+a*a*v+.2*u[...,0],b*a-a*a*v+.2*u[...,1]],axis=-1)
def step(x,u,family,changed):
    a=field(x,u,family,changed);b=field(x+.5*DT*a,u,family,changed);c=field(x+.5*DT*b,u,family,changed);d=field(x+DT*c,u,family,changed)
    return x+DT*(a+2*b+2*c+d)/6
def generate(family,seed,namespace,paths,length,changed=False,paired=False):
    r=np.random.default_rng(np.random.SeedSequence([seed,namespace]));nu=1 if family=='duffing' else 2
    init=r.uniform(-.8,.8,(paths,2)) if family=='duffing' else r.uniform([.5,1.],[1.5,2.],(paths,2))
    u=np.repeat(r.uniform(-1,1,(paths,(length+7)//8,nu)),8,axis=1)[:,:length]
    if paired:
        other=u.copy();other[:,32:64,0]=np.clip(other[:,32:64,0]+.5,-1,1)
        u=np.stack([u,other],axis=1).reshape(paths*2,length,nu);init=np.repeat(init,2,axis=0)
    x=[init]
    for t in range(length):x.append(step(x[-1],u[:,t],family,changed))
    x=np.stack(x,axis=1);assert np.isfinite(x).all() and np.max(np.abs(x))<1e4
    return x,u
def observe(s,changed):
    if not changed:return s.copy()
    rotation=np.array([[np.cos(.5),-np.sin(.5)],[np.sin(.5),np.cos(.5)]])
    return (s+.15*np.tanh(s))@(rotation@np.diag([.8,1.2])).T+np.array([.2,-.15])
def target_data(family,seed,mu,scale,case):
    dyn=case in ['dynamics','joint'];obs=case in ['observation','joint']
    a,u=generate(family,seed,3000,4,64,dyn);b,v=generate(family,seed,5000,2,64,dyn)
    rng=np.random.default_rng(np.random.SeedSequence([seed,4000]));anchors=rng.uniform(-1.5,1.5,(8,2));ya=observe(anchors,obs)+rng.normal(0,.01,anchors.shape)
    r=np.random.default_rng(np.random.SeedSequence([seed,7000]));y=observe((a-mu)/scale,obs)+r.normal(0,.01,a.shape);yv=observe((b-mu)/scale,obs)+r.normal(0,.01,b.shape)
    return dict(y=y,u=u,val_y=yv,val_u=v,anchor_source=anchors,anchor_target=ya)
def tensor(x):return torch.as_tensor(x,dtype=torch.float32)
def pack(agent):return {n:{k:v.detach().cpu().clone() for k,v in getattr(agent,n).state_dict().items()} for n in ['obs_enc','obs_dec','fwd_dyn','inv_dyn']}
def unpack(agent,state):
    for n,z in state.items():getattr(agent,n).load_state_dict(z)
def trainable(agent,names):
    for m in agent.modules:
        m.eval()
        for p in m.parameters():p.requires_grad_(False)
    params=[]
    for n in names:
        m=getattr(agent,n);m.train()
        for p in m.parameters():p.requires_grad_(True);params.append(p)
    return params
def freeze(agent):trainable(agent,[])
def predict(agent,y0,u,forward=None):
    with torch.no_grad():
        z=agent.obs_enc(tensor(y0));out=[];f=agent.fwd_dyn if forward is None else forward
        for t in range(u.shape[1]):z=f(torch.cat([z,tensor(u[:,t])],dim=-1));out.append(agent.obs_dec(z).numpy())
    return np.stack(out,axis=1).astype(np.float64)
def nrmse(pred,truth,h):
    if not np.isfinite(pred[:,:h]).all() or np.max(np.abs(pred[:,:h]))>1e4:return float('inf')
    return float(np.mean(np.sqrt(np.mean((pred[:,:h]-truth[:,1:h+1])**2,axis=1))/np.maximum(np.std(truth[:,1:h+1],axis=1),.05)))
def finite_score(score):return score if np.isfinite(score) else 1e12
def batch_windows(y,u,rng,batch,h=8):
    paths=rng.integers(0,len(y),size=batch);times=rng.integers(0,u.shape[1]-h+1,size=batch)
    return tensor(y[paths[:,None],times[:,None]+np.arange(h+1)]),tensor(u[paths[:,None],times[:,None]+np.arange(h)])
def output_loss(agent,y,u):
    z=agent.obs_enc(y[:,0]);errs=[]
    for t in range(u.shape[1]):z=agent.fwd_dyn(torch.cat([z,u[:,t]],dim=-1));errs.append(F.mse_loss(agent.obs_dec(z),y[:,t+1]))
    recon=F.mse_loss(agent.obs_dec(agent.obs_enc(y[:,0])),y[:,0])
    return torch.stack(errs).mean(),recon
class NativeLogger:
    def add_scalar(self,name,value,step):log(dict(kind='native_loss',name=name,value=float(value),step=int(step)))
