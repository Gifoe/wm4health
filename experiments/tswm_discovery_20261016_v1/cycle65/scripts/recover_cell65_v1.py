from models65 import *
import time,sys,faulthandler
faulthandler.enable(all_threads=True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest');torch.set_num_threads(4)
key,kind,attempt=sys.argv[1:4];resume=sys.argv[4] if len(sys.argv)>4 else None;mech,seed=key.split('_s');seed=int(seed);device=read(OUT/'QUALIFICATION65_V1.json')['device'];pp=paths();z=np.load(pp['data_root']/(key+'.npz'));train=z['train'];mean=train.mean((0,2),keepdims=True);std=train.std((0,2),keepdims=True);torch.manual_seed(seed);np.random.seed(seed);start=time.perf_counter();folder=(pp['checkpoint_root']/'recovery_v1'/key/kind/('attempt'+attempt)).resolve();folder.mkdir(exist_ok=False,parents=True);targets=0.;startstep=0
if kind.startswith('CSDI'):
 model=csdi(device);X=torch.tensor((train-mean)/std,device=device)
else:
 model=MDN().to(device);tx,ty=transitions(train);tx=torch.tensor(tx,device=device);ty=torch.tensor(ty,device=device)
opt=torch.optim.Adam(model.parameters(),lr=.001)
if resume:
 cp=torch.load(resume,map_location=device,weights_only=False);assert cp['key']==key and cp['kind']==kind;model.load_state_dict(cp['state_dict']);opt.load_state_dict(cp['optimizer']);startstep=cp['step'];targets=cp['target_scalar_exposures'];torch.set_rng_state(cp['rng_cpu'].cpu())
 if device=='cuda':torch.cuda.set_rng_state_all([r.cpu() for r in cp['rng_cuda']])
 nr=cp['rng_numpy'];np.random.set_state((nr['name'],np.array(nr['keys'],dtype='uint32'),nr['position'],nr['has_gauss'],nr['cached_gaussian']))
def cpu_tree(v):
 if isinstance(v,torch.Tensor):return v.detach().cpu().clone()
 if isinstance(v,dict):return {k:cpu_tree(x) for k,x in v.items()}
 if isinstance(v,list):return [cpu_tree(x) for x in v]
 if isinstance(v,tuple):return tuple(cpu_tree(x) for x in v)
 return v
for step in range(startstep+1,8001):
 model.train();opt.zero_grad()
 if kind.startswith('CSDI'):
  x=X[torch.randint(len(X),(64,),device=device)];draw=np.random.random() if kind.endswith('coverage') else 0.;m=model.get_randmask(torch.ones_like(x)) if draw<.5 else mask(x,'block' if draw<.75 else 'prefix');loss=model.calc_loss(x,m,torch.ones_like(m),side(model,x,m),1);targets+=float((1-m).sum())
 else:
  ids=torch.randint(len(tx),(512,),device=device);loss=model.nll(tx[ids],ty[ids]);targets+=512
 assert torch.isfinite(loss);loss.backward();opt.step()
 if step%500==0:state(f'recovery_{key}_{kind}_attempt{attempt}_state.json',dict(status='TRAINING',step=step,start_step=startstep,attempt=int(attempt),loss=float(loss)));print(json.dumps(dict(key=key,kind=kind,step=step,attempt=int(attempt))),flush=True)
 if step%2000==0:
  nr=np.random.get_state();checkpoint=dict(state_dict=cpu_tree(model.state_dict()),optimizer=cpu_tree(opt.state_dict()),step=step,key=key,kind=kind,mechanism=mech,seed=seed,regime=kind.split('_')[-1],normalization_mean=mean.tolist(),normalization_std=std.tolist(),config=config(),target_scalar_exposures=targets,rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all() if device=='cuda' else [],rng_numpy=dict(name=nr[0],keys=nr[1].tolist(),position=nr[2],has_gauss=nr[3],cached_gaussian=nr[4]),parameters=sum(v.numel() for v in model.parameters()),elapsed_seconds=time.perf_counter()-start,attempt=int(attempt),resumed_from=resume,source_receipt_sha256=sha(OUT/'SOURCES65_V1.json'));p=folder/f'step{step}.pt';tmp=p.with_suffix('.pt.tmp');assert p.resolve().parent==folder and tmp.resolve().parent==folder and not p.exists() and not tmp.exists();torch.save(checkpoint,tmp);tmp.replace(p);loaded=torch.load(p,map_location='cpu',weights_only=False);assert all(torch.equal(v,loaded['state_dict'][k]) for k,v in checkpoint['state_dict'].items());state(f'recovery_checkpoint_{key}_{kind}.json',dict(status='SEALED_BEFORE_VALIDATION',step=step,path=str(p),sha256=sha(p)))
state(f'recovery_{key}_{kind}_attempt{attempt}_state.json',dict(status='TRAINING_CHILD_CLOSED',step=8000,start_step=startstep,attempt=int(attempt),elapsed_seconds=time.perf_counter()-start));print(json.dumps(dict(status='TRAINING_CHILD_CLOSED',key=key,kind=kind)),flush=True)
