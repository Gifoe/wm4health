"""Does a post-hoc variance head recover a driver omitted by mean compression?

New controlled time-series experiment, NOT a VJEPA benchmark reproduction.
Matched architecture/data/updates: only uncertainty-to-encoder gradient differs.
Joint heteroscedastic learning is an existing-method control, not a new method.
"""
import datetime,hashlib,json,pathlib,time
import numpy as np
import torch
from torch import nn
ROOT=pathlib.Path(__file__).resolve().parents[1];DEVICE='cuda';D=32;STEPS=8000
def dump(p,o):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(o,indent=2),encoding='utf-8')
def generate(q,n,seed):
 g=torch.Generator(device=DEVICE);g.manual_seed(seed)
 def normal(*shape):return torch.randn(*shape,generator=g,device=DEVICE)
 def action():return torch.randint(0,2,(n,1),generator=g,device=DEVICE).float()*2-1
 y=normal(n,1);v=normal(n,1)
 for _ in range(100):
  a=action();y=.8*y+.3*a+(.27*(1+.8*a*v.tanh())).sqrt()*normal(n,1);v=normal(n,1)
 a=action();var=.27*(1+.8*a*v.tanh());mu=.8*y+.3*a
 yn=mu+var.sqrt()*normal(n,1);vn=normal(n,1)
 x=torch.cat([y,v,normal(n,D-2)],1)@q.T
 xn=torch.cat([yn,vn,normal(n,D-2)],1)@q.T
 return {'x':x,'future':xn,'action':a,'physical':torch.cat([y,v],1),'mean':mu,'variance':var}
class Model(nn.Module):
 def __init__(self,mode):
  super().__init__();self.mode=mode;self.enc=nn.Identity() if mode=='full_observation' else nn.Linear(D,2,bias=False)
  k=D if mode=='full_observation' else 2
  self.mean=nn.Linear(k+1,1);self.logvariance=nn.Sequential(nn.Linear(k+1,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,1))
 def forward(self,x,a):
  z=self.enc(x);features=torch.cat([z,a],1);mu=self.mean(features)
  variance_features=torch.cat([z.detach() if self.mode=='detached' else z,a],1)
  return mu,self.logvariance(variance_features).clamp(-6,2),z
def evaluate(model,probe,dev,direction,q):
 model.eval()
 with torch.no_grad():
  x=dev['x'];a=dev['action'];r=dev['future']@direction
  mu,lv,z=model(x,a);variance=lv.exp();alpha=(direction.T@q[:,:1]).item()
  true_mu=alpha*dev['mean'];true_var=alpha**2*dev['variance']+1-alpha**2
  kl=.5*(lv-true_var.log()+(true_var+(true_mu-mu).square())/variance-1)
  zp=model.enc(probe['x']).double();zp=torch.cat([zp,torch.ones(len(zp),1,device=DEVICE,dtype=torch.double)],1)
  coeff=torch.linalg.lstsq(zp,probe['physical'].double()).solution
  zd=torch.cat([z.double(),torch.ones(len(z),1,device=DEVICE,dtype=torch.double)],1)
  residual=zd@coeff-dev['physical'].double()
  nmse=residual.square().mean(0)/dev['physical'].double().var(0,unbiased=False)
  costs=[]
  for choice in [-1.,1.]:
   p,vp,_=model(x,torch.full_like(a,choice));costs.append(p.square()+vp.exp())
  chosen=torch.stack(costs,dim=-1).argmin(-1).float()*2-1
  yp=dev['physical'][:,:1];vv=dev['physical'][:,1:2]
  def cost(c):return (.8*yp+.3*c).square()+.27*(1+.8*c*vv.tanh())
  regret=cost(chosen)-torch.minimum(cost(-1),cost(1))
  def tail_prob(m,var):
   std=var.sqrt();cdf=lambda u:.5*(1+torch.erf(u/2**.5))
   return 1-cdf((1.5-m)/std)+cdf((-1.5-m)/std)
  event=(r.abs()>1.5).float();p=tail_prob(mu,variance);oracle=tail_prob(true_mu,true_var)
  return {'conditional_gaussian_kl':kl.mean().item(),'conditional_mean_mse':(mu-true_mu).square().mean().item(),'physical_y_probe_nmse':nmse[0].item(),'risk_v_probe_nmse':nmse[1].item(),'policy_regret':regret.mean().item(),'wrong_action_fraction':(regret>1e-7).float().mean().item(),'tail_brier':(p-event).square().mean().item(),'oracle_tail_brier':(oracle-event).square().mean().item(),'nll_without_constant':(.5*(lv+(r-mu).square()/variance)).mean().item()}
def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 out=ROOT/'outputs'/'risk_state_pilot';out.mkdir(parents=True,exist_ok=True)
 plan={'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'New controlled stochastic time-series diagnostic; no native-paper reproduction claim. Existing heteroscedastic likelihood is repair control.','question':'Does detaching an uncertainty head from a compressed mean-trained state erase information about action-dependent volatility despite accurate conditional means?','modes':['detached','attached','full_observation'],'initialization_seeds':[41,42,43],'data_seeds':{'train':10051,'probe':10052,'development':10053},'same_cohort_across_initializations':True,'training_pairs':16384,'burn_in':100,'dynamics':'y_next=.8*y+.3*a+sqrt(.27*(1+.8*a*tanh(v)))*eta; v_next iid N(0,1); a randomized +/-1; 30 iid observation nuisance coordinates; fixed unknown orthogonal observation map','allowed_information':'current full observation and proposed action; no hidden states or future observations at prediction','target_projection':'direction estimated from training-only OLS action coefficient in x_next ~ x_current + action; no oracle channel direction','encoder':'linear32->2 shared exact shape in detached/attached; full-observation informational control has no bottleneck and is not a capacity-matched competitor','heads':'linear conditional mean, two64-wide tanh conditional log-variance layers; same parameters for matched two variants','loss':'MSE(mean, projected future) +0.1 Gaussian NLL with residual mean detached in both; variance term input z is detached only in detached variant. Covariance may therefore teach attached encoder; existing heteroscedastic auxiliary objective, not proposed invention.','training':{'steps':STEPS,'batch':256,'optimizer':'AdamW','lr':.001,'weight_decay':.0001,'checkpoint':'fixed step8000; no hidden-state or policy-based selection'},'gate':'Exploratory support only if attached-minus-detached variance-driver NMSE improves by>.25, Gaussian KL by>.05, policy regret by>30%, and conditional mean MSE worsens by<.01 in all three training initializations. This is not candidate acceptance; independent data and native architectures/strong literature baselines required.'}
 path=out/'plan.json'
 if path.exists():assert json.loads(path.read_text())['training']==plan['training']
 else:dump(path,plan)
 g=torch.Generator();g.manual_seed(10001);q=torch.linalg.qr(torch.randn(D,D,generator=g)).Q.to(DEVICE)
 np.save(out/'observation_rotation.npy',q.cpu().numpy())
 train=generate(q,16384,10051);probe=generate(q,4096,10052);dev=generate(q,16384,10053)
 design=torch.cat([train['x'],train['action'],torch.ones(16384,1,device=DEVICE)],1).double()
 beta=torch.linalg.lstsq(design,train['future'].double()).solution
 direction=beta[-2].float().unsqueeze(1);direction=direction/direction.norm()
 np.save(out/'training_target_direction.npy',direction.cpu().numpy())
 dump(out/'target_direction_audit.json',{'fitted_on_hidden_state':False,'fitted_on_training_only':True,'squared_alignment_with_physical_output':((direction.T@q[:,:1])**2).item()})
 rows=[]
 for seed in plan['initialization_seeds']:
  for mode in plan['modes']:
   run=out/f'{mode}_seed{seed}';run.mkdir(exist_ok=True)
   if (run/'result.json').exists():rows.append(json.loads((run/'result.json').read_text()));continue
   torch.manual_seed(seed);model=Model(mode).to(DEVICE);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
   sampler=torch.Generator(device=DEVICE);sampler.manual_seed(10061);start=time.perf_counter()
   with (run/'train.jsonl').open('w',encoding='utf-8') as log:
    for step in range(1,STEPS+1):
     idx=torch.randint(0,16384,(256,),generator=sampler,device=DEVICE)
     target=train['future'][idx]@direction;mu,lv,z=model(train['x'][idx],train['action'][idx])
     mse=(mu-target).square().mean();nll=.5*(lv+(target-mu.detach()).square()/lv.exp()).mean();loss=mse+.1*nll
     opt.zero_grad(set_to_none=True);loss.backward();assert torch.isfinite(loss);opt.step()
     if step==1 or step%1000==0:
      log.write(json.dumps({'step':step,'loss':loss.item(),'mse':mse.item(),'nll':nll.item(),'seconds':time.perf_counter()-start})+'\n');log.flush()
      dump(out/'status.json',{'state':'training','seed':seed,'mode':mode,'step':step,'completed':len(rows),'total':9})
   torch.save(model.state_dict(),run/'model.pt')
   result={'seed':seed,'mode':mode,'parameters':sum(p.numel() for p in model.parameters()),'seconds':time.perf_counter()-start,'checkpoint_sha256':hashlib.sha256((run/'model.pt').read_bytes()).hexdigest(),**evaluate(model,probe,dev,direction,q)}
   dump(run/'result.json',result);rows.append(result);dump(out/'results.json',rows);print(json.dumps(result),flush=True)
 dump(out/'status.json',{'state':'complete','completed':len(rows),'total':9,'candidate_accepted':False})
if __name__=='__main__':main()
