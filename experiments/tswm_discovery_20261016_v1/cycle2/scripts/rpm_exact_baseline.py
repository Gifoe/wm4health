"""Existing Gaussian RPM with numerical marginal-likelihood training.

The positive-site variant keeps the native model's normalized full recognition
family. The full-Gaussian variant allows recognition variance above prior.
Both retain F in the objective and use ordinary finite-grid forward inference.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse,hashlib,json,pathlib,time
import numpy as np
import torch
from torch import nn
from generative_alarm_baseline import ROOT,grid_terms
torch.set_num_threads(4)

class Recognition(nn.Module):
 def __init__(self,kind,widths=(32,32)):
  super().__init__();layers=[];n=3
  for w in widths:layers.extend([nn.Linear(n,w),nn.ReLU()]);n=w
  layers.append(nn.Linear(n,2));self.network=nn.Sequential(*layers);self.kind=kind
 def natural(self,obs):
  raw=self.network(obs)
  if self.kind=='positive_site':p=1+torch.nn.functional.softplus(raw[...,1]).clamp_max(399.);b=raw[...,0]
  else:
   logv=raw[...,1].clamp(np.log(.0025),np.log(100.));p=torch.exp(-logv);b=raw[...,0]*p
  return p,b
 def logf(self,obs,grid):
  p,b=self.natural(obs)
  return .5*torch.log(p[...,None]/(2*np.pi))-.5*(p[...,None]*grid**2-2*b[...,None]*grid+b[...,None]**2/p[...,None])

def ratio_loglik(model,obs,grid,transition,initial,reference=None):
 f=model.logf(obs,grid);rf=f if reference is None else model.logf(reference,grid)
 F=torch.logsumexp(rf,dim=0)-np.log(len(rf));ll=f-F
 alpha=initial[None,:].expand(len(obs),-1);total=torch.zeros(len(obs),device=obs.device,dtype=obs.dtype)
 for t in range(obs.shape[1]):
  if t:alpha=alpha@transition
  offset=ll[:,t].max(dim=1,keepdim=True).values;alpha=alpha*torch.exp(ll[:,t]-offset);mass=alpha.sum(dim=1,keepdim=True).clamp_min(torch.finfo(obs.dtype).tiny)
  total=total+torch.log(mass[:,0])+offset[:,0];alpha=alpha/mass
 return total

def train():
 pp=ROOT/'outputs/rpm_exact_training_plan.json';plan=json.loads(pp.read_text());out=ROOT/'outputs/rpm_exact';out.mkdir(exist_ok=True,parents=True)
 assert torch.cuda.is_available();device='cuda';grid,K,initial=grid_terms(plan['train_grid']['points'],plan['train_grid']['extent'],plan['rho'],device)
 for case in plan['cases']:
  kind=case['dataset'];variant=case['recognition'];data=ROOT/'outputs/native_alarm'/(kind+'_data.npz')
  with np.load(data) as f:train_x=torch.tensor(f['train_obs'],device=device,dtype=torch.float32);val_x=torch.tensor(f['val_obs'],device=device,dtype=torch.float32)
  for seed in case['seeds']:
   stem=kind+'_'+variant+'_seed'+str(seed);result_path=out/(stem+'.json')
   if result_path.exists():print('retained completed result',stem,flush=True);continue
   torch.manual_seed(seed);rng=np.random.default_rng(seed);model=Recognition(variant,plan['hidden_widths']).to(device);optimizer=torch.optim.Adam(model.parameters(),lr=plan['learning_rate']);start=time.perf_counter();rows=[];best={'loss':float('inf'),'iteration':None,'weights':None}
   for iteration in range(1,plan['training_iterations']+1):
    ids=torch.tensor(rng.integers(0,len(train_x),plan['batch_size']),device=device);batch=train_x[ids]
    optimizer.zero_grad();loss=-ratio_loglik(model,batch,grid,K,initial).mean()/train_x.shape[1];assert torch.isfinite(loss);loss.backward();optimizer.step()
    if iteration==1 or iteration%200==0:
     with torch.no_grad():vl=float(torch.stack([-ratio_loglik(model,val_x[i:i+64],grid,K,initial).mean()/val_x.shape[1] for i in range(0,len(val_x),64)]).mean())
     row={'dataset':kind,'recognition':variant,'seed':seed,'iteration':iteration,'train_negative_association_loglik':float(loss),'validation_negative_association_loglik':vl,'elapsed_seconds':time.perf_counter()-start};rows.append(row)
     if vl<best['loss']:best={'loss':vl,'iteration':iteration,'weights':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}}
     with (out/'training.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
     print(json.dumps(row),flush=True)
   path=out/(stem+'.pt');torch.save(best['weights'],path)
   result={'dataset':kind,'recognition':variant,'seed':seed,'iterations_completed':plan['training_iterations'],'selected_iteration':best['iteration'],'selected_validation_negative_association_loglik':best['loss'],
    'parameters':sum(v.numel() for v in model.parameters()),'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'plan_sha256':hashlib.sha256(pp.read_bytes()).hexdigest(),'code_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
    'data_sha256':hashlib.sha256(data.read_bytes()).hexdigest(),'torch_version':torch.__version__,'device':torch.cuda.get_device_name(),'runtime_seconds':time.perf_counter()-start,'training_rows':rows}
   result_path.write_text(json.dumps(result,indent=2),encoding='utf-8')

def evaluate():
 from evaluate_native_alarm import filter_grid,mixture_logs,moments,readout,summarize
 from threadpoolctl import threadpool_limits
 pp=ROOT/'outputs/rpm_exact_training_plan.json';plan=json.loads(pp.read_text());baseplan=json.loads((ROOT/'outputs/native_learning_plan.json').read_text());out=ROOT/'outputs/rpm_exact';grid=np.linspace(-8,8,401);finegrid=np.linspace(-10,10,801);prefixes=baseplan['prefix_lengths']
 with threadpool_limits(limits=4):
  for case in plan['cases']:
   kind=case['dataset'];variant=case['recognition']
   with np.load(ROOT/'outputs/native_alarm'/(kind+'_data.npz')) as f:arrays={split:{k:f[split+'_'+k] for k in ['obs','z','mask','alarm']} for split in ['train','test']}
   for seed in case['seeds']:
    stem=kind+'_'+variant+'_seed'+str(seed);result_path=out/(stem+'_evaluation.json')
    if result_path.exists():print('retained evaluation',stem,flush=True);continue
    if not (out/(stem+'.json')).exists():print('pending training',stem,flush=True);continue
    model=Recognition(variant,plan['hidden_widths']);cp=out/(stem+'.pt');model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True));model.eval()
    def stats(obs):
     with torch.no_grad():p,b=model.natural(torch.tensor(obs,dtype=torch.float32));return p.numpy().astype('float64')-1,b.numpy().astype('float64')
    train_p,train_b=stats(arrays['train']['obs']);test_p,test_b=stats(arrays['test']['obs']);F=mixture_logs(train_p,train_b,grid)
    cal={k:v[:1024] for k,v in arrays['train'].items()};calpost=filter_grid(baseplan,kind,cal,grid,[16],(train_p[:1024],train_b[:1024]),F)[16];C,bias=readout(moments(calpost,grid)[0],cal['z'][:,-1])
    test=filter_grid(baseplan,kind,arrays['test'],grid,prefixes,(test_p,test_b),F);rows=[];raw={}
    for L in prefixes:
     m,v=moments(test[L],grid);raw[str(L)+'_mean']=m;raw[str(L)+'_variance']=v
     for row in summarize(baseplan,arrays['test'],L,m,v,test[L],grid,(C,bias),raw,variant+'_exact_ML'):row['method']=variant+'_exact_ML';rows.append(row)
    small={k:v[:64] for k,v in arrays['test'].items()};fineF=mixture_logs(train_p,train_b,finegrid);fine=filter_grid(baseplan,kind,small,finegrid,prefixes,(test_p[:64],test_b[:64]),fineF);checks=[]
    for L in prefixes:
     mf,vf=moments(fine[L],finegrid);mc,vc=moments(test[L][:64],grid);checks.append({'prefix':L,'max_mean_difference':float(abs(mf-mc).max()),'max_variance_difference':float(abs(vf-vc).max())})
    assert max(v['max_mean_difference'] for v in checks)<1e-4
    assert max(v['max_variance_difference'] for v in checks)<1e-4
    rawpath=out/(stem+'_evaluation_arrays.npz');np.savez_compressed(rawpath,**raw)
    result={'dataset':kind,'seed':seed,'recognition':variant,'readout':{'slope':C,'intercept':bias},'metrics':rows,'grid_checks':checks,'plan_sha256':hashlib.sha256(pp.read_bytes()).hexdigest(),'checkpoint_sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),'raw_arrays_sha256':hashlib.sha256(rawpath.read_bytes()).hexdigest(),
     'scope':'Existing model/objective numerical baseline; outside empirical atomic support this is a training-reference plug-in ratio posterior, not an exact continuous observation likelihood.'}
    result_path.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({'dataset':kind,'recognition':variant,'seed':seed,'readout':result['readout'],'primary':[r for r in rows if r['prefix']==16 and r['horizon']==1 and r['subset']=='alarm_no_anchor']}),flush=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['train','evaluate']);args=parser.parse_args();train() if args.mode=='train' else evaluate()
