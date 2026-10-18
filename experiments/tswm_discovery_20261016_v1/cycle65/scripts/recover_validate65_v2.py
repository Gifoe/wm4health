from models65 import *
from checkpoint_io65_v1 import load_checkpoint
import sys,time,faulthandler
faulthandler.enable(all_threads=True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest');torch.set_num_threads(4)
key,kind=sys.argv[1:3];mech,seed=key.split('_s');seed=int(seed);device=read(OUT/'QUALIFICATION65_V1.json')['device'];pp=paths();datafile=pp['data_root']/(key+'.npz');raw=np.load(datafile);checkpoints=[];payload8=None
for step in [2000,8000]:
 candidates=list((pp['checkpoint_root']/'recovery_v1'/key/kind).glob(f'attempt*/step{step}.pt'))+list((pp['checkpoint_root']/'recovery_npz_v1'/key/kind).glob(f'attempt*/step{step}.npz'));candidates=sorted(candidates,key=lambda p:p.stat().st_mtime);assert candidates;path=candidates[-1];z=load_checkpoint(path,map_location=device);assert z['step']==step and all(torch.isfinite(v).all() for v in z['state_dict'].values());start=time.perf_counter()
 if kind.startswith('CSDI'):
  model=csdi(device);model.load_state_dict(z['state_dict']);model.eval();mean=np.asarray(z['normalization_mean'],dtype='float32');std=np.asarray(z['normalization_std'],dtype='float32');x=torch.tensor((raw['validation'][:256]-mean)/std,device=device);values={}
  with torch.no_grad(),torch.random.fork_rng(devices=[0] if device=='cuda' else []):
   for view in ['block','prefix']:
    torch.manual_seed(650003);m=mask(x,view,np.random.default_rng(650004));values[view]=float(model.calc_loss_valid(x,m,torch.ones_like(m),side(model,x,m),0))
  values['mean']=sum(values.values())/2;cp=dict(path=str(path),sha256=sha(path),step=step,validation=values)
 else:
  model=MDN().to(device);model.load_state_dict(z['state_dict']);model.eval();vx,vy=transitions(raw['validation'])
  with torch.no_grad():nll=float(model.nll(torch.tensor(vx,device=device),torch.tensor(vy,device=device)))
  cp=dict(path=str(path),sha256=sha(path),step=step,validation_NLL=nll)
 checkpoints.append(cp)
 if step==8000:payload8={k:z[k] for k in ['parameters','target_scalar_exposures','elapsed_seconds']}
 del model,z
selected=min(checkpoints,key=lambda c:c['validation']['mean'] if kind.startswith('CSDI') else c['validation_NLL']);fit=dict(key=key,model=kind,mechanism=mech,seed=seed,parameters=payload8['parameters'],optimizer_updates=8000,target_scalar_exposures=payload8['target_scalar_exposures'],elapsed_seconds=payload8['elapsed_seconds'],checkpoints=checkpoints,selected_checkpoint=selected,data_sha256=sha(datafile),selection='fixed block/prefix validation DSM only' if kind.startswith('CSDI') else 'validation one-step conditional NLL only',execution_variant='Isolated recovery process; CPU snapshots atomically before separate native validation; model/loss/budget unchanged')
if kind.startswith('CSDI'):fit['trajectory_presentations']=8000*64
dump(OUT/f'RECOVERED_FIT_{key}_{kind}_V1.json',fit);print(json.dumps(dict(status='RECOVERED_FIT_QUALIFIED',key=key,kind=kind,selected_step=selected['step'])))
