"""Ordinary nonlinear-emission SSM baseline with discrete quadrature ML.

No hidden-state labels, oracle alarm function, reweighting or proposed new
algorithm. The known correct AR prior matches the native mechanism diagnostic.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse,hashlib,json,pathlib,time
import numpy as np
from scipy.stats import norm
import torch
from torch import nn
ROOT=pathlib.Path(__file__).resolve().parents[1]
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False

class ObservationModel(nn.Module):
 def __init__(self,widths):
  super().__init__();layers=[];last=1
  for w in widths:layers.extend([nn.Linear(last,w),nn.ReLU()]);last=w
  layers.append(nn.Linear(last,1));self.alarm=nn.Sequential(*layers)
  self.anchor_scale=nn.Parameter(torch.tensor(.8));self.anchor_bias=nn.Parameter(torch.tensor(0.));self.log_noise=nn.Parameter(torch.tensor(np.log(.5),dtype=torch.float32))
 def likelihood_grid(self,grid):
  return self.anchor_scale*grid+self.anchor_bias,self.alarm(grid[:,None])[:,0],self.log_noise.exp().clamp_min(.01)

def loglik(model,obs,grid,transition,initial):
 mean,logit,sd=model.likelihood_grid(grid);x=obs[:,:,0];mask=obs[:,:,1];alarm=obs[:,:,2]
 ll=-.5*((x[:,:,None]-mean)/sd)**2-torch.log(sd)-.5*np.log(2*np.pi)
 ll=ll*mask[:,:,None]+alarm[:,:,None]*torch.nn.functional.logsigmoid(logit)+(1-alarm[:,:,None])*torch.nn.functional.logsigmoid(-logit)
 alpha=initial[None,:].expand(len(obs),-1);tot=torch.zeros(len(obs),device=obs.device,dtype=obs.dtype)
 for t in range(obs.shape[1]):
  if t:alpha=alpha@transition
  offset=ll[:,t].max(dim=1,keepdim=True).values
  alpha=alpha*torch.exp(ll[:,t]-offset);mass=alpha.sum(dim=1,keepdim=True).clamp_min(torch.finfo(obs.dtype).tiny)
  tot=tot+torch.log(mass[:,0])+offset[:,0];alpha=alpha/mass
 return tot

def grid_terms(points,extent,rho,device,dtype=torch.float32):
 grid=torch.linspace(-extent,extent,points,device=device,dtype=dtype);step=float(2*extent/(points-1));sd=np.sqrt(1-rho*rho)
 transition=torch.exp(-.5*((grid[None,:]-rho*grid[:,None])/sd)**2)/(np.sqrt(2*np.pi)*sd)*step
 initial=torch.exp(-.5*grid*grid)/np.sqrt(2*np.pi)*step;initial=initial/initial.sum()
 return grid,transition,initial

def train():
 pp=ROOT/'outputs/generative_alarm_plan.json';plan=json.loads(pp.read_text());out=ROOT/'outputs/generative_alarm';out.mkdir(exist_ok=True,parents=True)
 assert torch.cuda.is_available(),'This baseline was planned for the available GPU'
 device=torch.device('cuda');grid,transition,initial=grid_terms(plan['train_grid']['points'],plan['train_grid']['extent'],.95,device)
 for kind in plan['datasets']:
  data=ROOT/'outputs/native_alarm'/(kind+'_data.npz')
  with np.load(data) as f:train_x=torch.tensor(f['train_obs'],device=device,dtype=torch.float32);val_x=torch.tensor(f['val_obs'],device=device,dtype=torch.float32)
  for seed in plan['training_seeds']:
   stem=kind+'_seed'+str(seed);result_file=out/(stem+'.json')
   if result_file.exists():print('retained completed result',stem,flush=True);continue
   torch.manual_seed(seed);rng=np.random.default_rng(seed);model=ObservationModel(plan['alarm_hidden_widths']).to(device);optimizer=torch.optim.Adam(model.parameters(),lr=plan['learning_rate'])
   best={'nll':float('inf'),'iteration':None,'weights':None};start=time.perf_counter();rows=[]
   for iteration in range(1,plan['training_iterations']+1):
    indices=torch.tensor(rng.integers(0,len(train_x),size=plan['batch_size']),device=device);batch=train_x[indices]
    optimizer.zero_grad();loss=-loglik(model,batch,grid,transition,initial).mean()/train_x.shape[1]
    assert torch.isfinite(loss);loss.backward();optimizer.step()
    if iteration==1 or iteration%200==0:
     with torch.no_grad():vl=float(-loglik(model,val_x,grid,transition,initial).mean()/val_x.shape[1])
     row={'dataset':kind,'seed':seed,'iteration':iteration,'train_nll':float(loss),'validation_nll':vl,'elapsed_seconds':time.perf_counter()-start};rows.append(row)
     if vl<best['nll']:best={'nll':vl,'iteration':iteration,'weights':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}}
     with (out/'training.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
     print(json.dumps(row),flush=True)
   checkpoint=out/(stem+'.pt');torch.save(best['weights'],checkpoint)
   result={'dataset':kind,'seed':seed,'iterations_completed':plan['training_iterations'],'selected_iteration':best['iteration'],'selected_validation_nll':best['nll'],
    'parameters':sum(p.numel() for p in model.parameters()),'plan_sha256':hashlib.sha256(pp.read_bytes()).hexdigest(),'code_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
    'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'data_sha256':hashlib.sha256(data.read_bytes()).hexdigest(),'torch_version':torch.__version__,
    'device':torch.cuda.get_device_name(),'runtime_seconds':time.perf_counter()-start,'training_rows':rows}
   result_file.write_text(json.dumps(result,indent=2),encoding='utf-8')

def filter_learned(baseplan,arrays,model,grid,prefixes):
 with torch.no_grad():
  mean,logit,sd=model.likelihood_grid(torch.tensor(grid,dtype=torch.float32));mean=mean.numpy();logit=logit.numpy();sd=float(sd)
 rho=baseplan['rho'];step=grid[1]-grid[0];transition=norm.pdf(grid[None,:],loc=rho*grid[:,None],scale=np.sqrt(1-rho*rho))*step
 weights=np.tile(norm.pdf(grid),(len(arrays['obs']),1));weights/=weights.sum(axis=1,keepdims=True);outputs={}
 for t in range(arrays['obs'].shape[1]):
  if t:weights=weights@transition
  x=arrays['obs'][:,t,0];mask=arrays['mask'][:,t];alarm=arrays['alarm'][:,t]
  ll=-.5*((x[:,None]-mean[None,:])/sd)**2*mask[:,None]
  ll+=np.where(alarm[:,None],-np.logaddexp(0.,-logit),-np.logaddexp(0.,logit))
  weights*=np.exp(ll-ll.max(axis=1,keepdims=True));weights/=weights.sum(axis=1,keepdims=True)
  if t+1 in prefixes:outputs[t+1]=weights.copy()
 return outputs

def evaluate():
 from evaluate_native_alarm import moments,readout,summarize
 from threadpoolctl import threadpool_limits
 pp=ROOT/'outputs/generative_alarm_plan.json';plan=json.loads(pp.read_text());bp=ROOT/'outputs/native_learning_plan.json';baseplan=json.loads(bp.read_text());out=ROOT/'outputs/generative_alarm';start=time.perf_counter()
 grid=np.linspace(-8,8,401);finegrid=np.linspace(-10,10,801);prefixes=baseplan['prefix_lengths']
 with threadpool_limits(limits=4):
  for kind in plan['datasets']:
   with np.load(ROOT/'outputs/native_alarm'/(kind+'_data.npz')) as f:arrays={split:{k:f[split+'_'+k] for k in ['obs','z','mask','alarm']} for split in ['train','test']}
   for seed in plan['training_seeds']:
    stem=kind+'_seed'+str(seed);result_path=out/(stem+'_evaluation.json')
    if result_path.exists():print('retained evaluation',stem,flush=True);continue
    if not (out/(stem+'.json')).exists():print('pending training',stem,flush=True);continue
    model=ObservationModel(plan['alarm_hidden_widths']);checkpoint=out/(stem+'.pt');model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True));model.eval()
    cal={k:v[:1024] for k,v in arrays['train'].items()};cm=moments(filter_learned(baseplan,cal,model,grid,[16])[16],grid)[0];C,bias=readout(cm,cal['z'][:,-1])
    test=filter_learned(baseplan,arrays['test'],model,grid,prefixes);raw={};rows=[]
    for L in prefixes:
     m,v=moments(test[L],grid);raw[str(L)+'_mean']=m;raw[str(L)+'_variance']=v
     for row in summarize(baseplan,arrays['test'],L,m,v,test[L],grid,(C,bias),raw,'generative_ssm'):row['method']='generative_ssm';rows.append(row)
    small={k:v[:64] for k,v in arrays['test'].items()};fine=filter_learned(baseplan,small,model,finegrid,prefixes);checks=[]
    for L in prefixes:
     mf,vf=moments(fine[L],finegrid);mc,vc=moments(test[L][:64],grid);checks.append({'prefix':L,'max_mean_difference':float(abs(mf-mc).max()),'max_variance_difference':float(abs(vf-vc).max())})
    assert max(r['max_mean_difference'] for r in checks)<5e-4
    assert max(r['max_variance_difference'] for r in checks)<5e-4
    rawpath=out/(stem+'_evaluation_arrays.npz');np.savez_compressed(rawpath,**raw)
    result={'dataset':kind,'seed':seed,'method':'ordinary nonlinear-emission SSM ML','readout':{'slope':C,'intercept':bias},'metrics':rows,'grid_checks':checks,
     'learned_anchor_scale':float(model.anchor_scale),'learned_anchor_bias':float(model.anchor_bias),'learned_anchor_noise':float(model.log_noise.exp()),
     'plan_sha256':hashlib.sha256(pp.read_bytes()).hexdigest(),'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'raw_arrays_sha256':hashlib.sha256(rawpath.read_bytes()).hexdigest()}
    result_path.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({'dataset':kind,'seed':seed,'readout':result['readout'],'primary':[r for r in rows if r['prefix']==16 and r['horizon']==1 and r['subset']=='alarm_no_anchor'],'elapsed_seconds':time.perf_counter()-start}),flush=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['train','evaluate']);args=parser.parse_args();train() if args.mode=='train' else evaluate()
