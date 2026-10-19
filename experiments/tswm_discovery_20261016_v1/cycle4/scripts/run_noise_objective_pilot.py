"""Controlled numerical objective audit, not native visual-benchmark reproduction.

Unmodified official JEPA/MotionJEPA train_step and SIGReg, with explicitly linear
vector encoders/predictor adapters. All training labels are observations/actions.
Hidden states are used only by a frozen diagnostic probe and analytic checks.
"""
import datetime,hashlib,json,pathlib,random,sys,time
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
ROOT=pathlib.Path(__file__).resolve().parents[1]
SOURCE=ROOT/'sources'/'motion-jepa'
sys.path.insert(0,str(SOURCE))
from src.models.lewm.jepa import JEPA
from src.models.motionjepa.jepa import MotionJEPA
from src.models.lewm.module import SIGReg
DEVICE='cuda'
DIM=32
LATENT=2
STEPS=2000
BATCH=128
SEED=41
class VectorEncoder(nn.Module):
 def __init__(self):super().__init__();self.map=nn.Linear(DIM,LATENT,bias=False)
 def forward(self,pixels,**kw):return SimpleNamespace(last_hidden_state=self.map(pixels).unsqueeze(1))
class CausalFIR(nn.Module):
 def __init__(self):
  super().__init__();self.map=nn.Linear(3*(LATENT+2),LATENT,bias=False)
 def forward(self,z,a):
  features=torch.cat([z,a],dim=-1);history=[]
  for lag in [2,1,0]:
   history.append(torch.nn.functional.pad(features,(0,0,lag,0))[:,:features.shape[1]])
  return self.map(torch.cat(history,dim=-1))
def dump(path,obj):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,indent=2),encoding='utf-8')
def seed_all(seed):random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
def make_data(q,n,t,regime,seed):
 g=torch.Generator(device=DEVICE);g.manual_seed(seed)
 a=torch.randn(n,t,2,generator=g,device=DEVICE)
 eta=torch.randn(n,t,2,generator=g,device=DEVICE)
 rho=torch.tensor([.95,.8],device=DEVICE);innovation=(1-rho*rho).sqrt()
 # 80% of innovation variance is explained by the supplied action; the rest
 # is process noise. Stationary physical variance is exactly one per coordinate.
 states=[torch.randn(n,2,generator=g,device=DEVICE)]
 for i in range(t-1):states.append(rho*states[-1]+innovation*(.8**.5*a[:,i]+.2**.5*eta[:,i]))
 states=torch.stack(states,dim=1)
 nuisance=torch.randn(n,t,DIM,generator=g,device=DEVICE)
 if regime=='clean':nuisance.zero_();nuisance[:,:,:2]=states
 elif regime=='independent_nuisance':nuisance[:,:,:2]=states
 elif regime=='sensor_noise':nuisance[:,:,:2]=.5*states+(3/4)**.5*nuisance[:,:,:2]
 else:raise ValueError(regime)
 x=nuisance@q.T
 return {'pixels':x,'action':a},states
def build(variant):
 common=dict(encoder=VectorEncoder(),predictor=CausalFIR(),action_encoder=nn.Identity(),embed_dim=LATENT,history_size=3,num_preds=1)
 if variant=='motion':model=MotionJEPA(diff_encoder=VectorEncoder(),**common)
 else:model=JEPA(sigreg_weight=.25 if variant=='lewm_matched_static' else .09,**common)
 return model.to(DEVICE)
def matrix_stats(x,s):
 x=x['pixels'].reshape(-1,DIM).double();s=s.reshape(-1,2).double()
 x=x-x.mean(0);s=s-s.mean(0)
 return {'covariance':(x.T@x/len(x)).cpu().tolist(),'physical_covariance':(s.T@s/len(x)).cpu().tolist()}
def probe(model,train,test,q):
 model.eval()
 with torch.no_grad():
  x,s=train;y,t=test
  z=model.encoder(x['pixels'].reshape(-1,DIM)).last_hidden_state[:,0].double();labels=s.reshape(-1,2).double()
  zt=model.encoder(y['pixels'].reshape(-1,DIM)).last_hidden_state[:,0].double();truth=t.reshape(-1,2).double()
  bias=torch.ones(len(z),1,device=DEVICE,dtype=torch.double);design=torch.cat([z,bias],1)
  beta=torch.linalg.solve(design.T@design+1e-6*torch.eye(3,device=DEVICE,dtype=torch.double),design.T@labels)
  residual=torch.cat([zt,torch.ones(len(zt),1,device=DEVICE,dtype=torch.double)],1)@beta-truth
  nmse=(residual.square().mean(0)/truth.var(0,unbiased=False)).mean().item()
  w=model.encoder.map.weight.double();u=q[:,:2].double();p=w.T@torch.linalg.pinv(w@w.T)@w
  overlap=torch.trace(u.T@p@u).item()/2
 return {'physical_probe_nmse':nmse,'signal_subspace_overlap':overlap,'encoder_covariance_eigenvalues':torch.linalg.eigvalsh(zt.T@zt/len(zt)).cpu().tolist()}
def linear_predictive_subspace(train,test,q):
 x,s=train;y,t=test
 flat=x['pixels'][:,:-1].reshape(-1,DIM).double();future=x['pixels'][:,1:].reshape(-1,DIM).double()
 mu=flat.mean(0);nu=future.mean(0);flat=flat-mu;future=future-nu
 c00=flat.T@flat/len(flat);c11=future.T@future/len(flat);c01=flat.T@future/len(flat)
 def inverse_root(c):
  d,v=torch.linalg.eigh(c);return (v*d.clamp_min(1e-5).rsqrt())@v.T
 a=inverse_root(c00);b=inverse_root(c11);u,sv,_=torch.linalg.svd(a@c01@b)
 w=(a@u[:,:2]).T
 model=build('lewm_matched_static');model.encoder.map.weight.data.copy_(w.float())
 result=probe(model,train,test,q);result.update(method='linear_CCA_VAMP_style',canonical_singular_values=sv[:6].cpu().tolist(),uses_hidden_states_to_fit_encoder=False)
 return result
def main():
 torch.set_num_threads(4);assert torch.cuda.is_available();torch.backends.cuda.matmul.allow_tf32=False
 out=ROOT/'outputs'/'noise_objective_pilot';out.mkdir(parents=True,exist_ok=True)
 plan={'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'question':'Does the native temporal-difference auxiliary suppress predictive physical directions in the presence of independent observation nuisance?','source_revision':'b3c2fb374b7c69f169de15774f371290a8736f56','scope':'New linear-Gaussian vector-adapter numerical objective audit; NOT native visual benchmark or original-paper reproduction','variants':['lewm_native_weight','lewm_matched_static','motion'],'regimes':['clean','independent_nuisance'],'training_initialization':SEED,'data_seeds':{'train':9051,'probe_train':9052,'development':9053},'dimension':DIM,'latent_dimension':LATENT,'physical_dynamics':{'rho':[.95,.8],'action_fraction_of_innovation_variance':.8,'process_fraction':.2},'train':{'steps':STEPS,'batch':BATCH,'optimizer':'AdamW','lr':.0005,'weight_decay':.001,'checkpoint':'fixed final step; no state-labelled checkpoint selection'},'native_functions':['JEPA.train_step','MotionJEPA.train_step','SIGReg default1024 random projections'],'adapter':'Linear per-frame observation/difference encoders; causal linear FIR predictor with3-step context; native 512-wide auxiliary inverse predictor retained','primary':'frozen held-out physical-state linear-probe NMSE and signal-subspace overlap','decision_gate':'Investigate a noise-specific failure only if motion-minus-matched-static NMSE >0.10 on independent_nuisance and <0.05 on clean; otherwise do not infer the proposed structural failure. This pilot cannot endorse an idea; requires independent seeds/data and full native architecture confirmation.','existing_method_control':'Ordinary linear CCA/VAMP-style predictive subspace; not claimed new','no_hypothesis_salvage_by_test_tuning':True}
 planpath=out/'plan.json'
 if planpath.exists():assert json.loads(planpath.read_text())['train']==plan['train']
 else:dump(planpath,plan)
 gen=torch.Generator();gen.manual_seed(9001);q=torch.linalg.qr(torch.randn(DIM,DIM,generator=gen)).Q.to(DEVICE)
 np.save(out/'observation_rotation.npy',q.cpu().numpy())
 rows=[]
 for regime in plan['regimes']:
  training,_=make_data(q,8192,4,regime,9051)
  prober=make_data(q,4096,4,regime,9052);dev=make_data(q,8192,4,regime,9053)
  dump(out/f'{regime}_data_covariance.json',matrix_stats(*prober))
  cca=linear_predictive_subspace(prober,dev,q);dump(out/f'{regime}_linear_control.json',cca)
  for variant in plan['variants']:
   run=out/f'{regime}_{variant}_seed{SEED}';run.mkdir(exist_ok=True)
   if (run/'result.json').exists():rows.append(json.loads((run/'result.json').read_text()));continue
   seed_all(SEED);model=build(variant);opt=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.001)
   # Sampling RNG is independent of SIGReg/model RNG. Every method receives the
   # exact same sample-index sequence, observations and actions in this regime.
   indices_rng=torch.Generator(device=DEVICE);indices_rng.manual_seed(9061)
   start=time.perf_counter()
   with (run/'train.jsonl').open('w',encoding='utf-8') as logfile:
    for step in range(1,STEPS+1):
     model.train();indices=torch.randint(0,8192,(BATCH,),generator=indices_rng,device=DEVICE)
     batch={k:v[indices] for k,v in training.items()};opt.zero_grad(set_to_none=True)
     _,losses=model.train_step(batch);losses['loss'].backward()
     assert torch.isfinite(losses['loss']) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
     opt.step()
     if step==1 or step%200==0:
      log={'step':step,**{k:float(v.detach()) for k,v in losses.items()},'seconds':time.perf_counter()-start};logfile.write(json.dumps(log)+'\n');logfile.flush()
      dump(out/'status.json',{'state':'training','regime':regime,'variant':variant,'step':step,'completed':len(rows),'total':6})
   torch.save(model.state_dict(),run/'model.pt');result={'regime':regime,'variant':variant,'seed':SEED,'steps':STEPS,'seconds':time.perf_counter()-start,'parameters':sum(p.numel() for p in model.parameters()),'checkpoint_sha256':hashlib.sha256((run/'model.pt').read_bytes()).hexdigest(),**probe(model,prober,dev,q)}
   dump(run/'result.json',result);rows.append(result);dump(out/'results.json',rows);print(json.dumps(result),flush=True)
   del model;torch.cuda.empty_cache()
  del training,prober,dev
 dump(out/'status.json',{'state':'complete','completed':len(rows),'total':6,'candidate_accepted':False})
if __name__=='__main__':main()
