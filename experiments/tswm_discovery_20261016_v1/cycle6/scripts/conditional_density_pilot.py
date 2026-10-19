"""Isolate native inverse-MLP likelihood validity from latent recovery/data bugs."""
import runtime_profile
import datetime,json,math,pathlib,time
import numpy as np,torch
from torch import nn
from torch.func import vmap,jacfwd
from native_child import ROOT,classes
from piecewise_density import audit,normalized_samples
def dump(p,o):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(o,indent=2),encoding='utf-8')
def true_sigma(c):return .3+.7*torch.sigmoid(c)
class GaussianControl(nn.Module):
 def __init__(self,native_cls):
  super().__init__();self.conditioner=native_cls(1,2,3,128)
 def logp(self,c,x):
  result=self.conditioner(c[:,None]);mu=result[:,0];sigma=torch.nn.functional.softplus(result[:,1])+1e-3
  return -.5*((x-mu)/sigma).square()-sigma.log()-.5*math.log(2*math.pi)
def inverse_logp(net,c,x):
 inputs=torch.stack([c,x],-1);e=net(inputs)[:,0];j=vmap(jacfwd(net))(inputs)[:,0,-1]
 return -.5*e.square()+j.abs().clamp_min(1e-30).log()-.5*math.log(2*math.pi)
def verify_auditor(out):
 from scipy.special import ndtr,ndtri
 class Net(nn.Module):
  def __init__(self,fold=False,bias=0.):
   super().__init__()
   if fold:
    self.net=nn.Sequential(nn.Linear(2,2),nn.LeakyReLU(.2),nn.Linear(2,1))
    with torch.no_grad():self.net[0].weight.copy_(torch.tensor([[0.,1.],[0.,-1.]]));self.net[0].bias.zero_();self.net[2].weight.fill_(1/.8);self.net[2].bias.fill_(bias)
   else:
    self.net=nn.Sequential(nn.Linear(2,1))
    with torch.no_grad():self.net[0].weight.copy_(torch.tensor([[0.,1.]]));self.net[0].bias.zero_()
 rows=[]
 for fold,bias in [(False,0.),(True,0.),(True,-1.)]:
  net=Net(fold,bias).double();result=audit(net,[0.]);expected=2*(1-ndtr(bias)) if fold else 1.;assert abs(result['integrated_surrogate_mass']-expected)<1e-12
  rows.append({'fold':fold,'bias':bias,'expected_mass':float(expected),**result})
 net=Net().double();u=np.linspace(.001,.999,1000);q=normalized_samples(net,[2.],u);assert np.max(np.abs(q-ndtri(u)))<1e-10
 dump(out/'auditor_validation.json',{'analytic_cases':rows,'identity_quantile_max_error':float(np.max(np.abs(q-ndtri(u))))})
def main():
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;out=ROOT/'outputs/conditional_density';out.mkdir(parents=True,exist_ok=True)
 if (out/'plan.json').exists():raise RuntimeError('Existing density pilot; inspect exact process before restart')
 plan={'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'New controlled component experiment, not a native paper-table reproduction or accepted research gap','seeds':[81,82,83],'cohort_seed':991,'training':8192,'validation':2048,'test_per_context':8192,'test_contexts':[-2.,0.,2.],'DGP':'x=.6*c+(.3+.7*sigmoid(c))*epsilon, c and epsilon independent standard normal; both inputs observed, no latent identification burden','steps':2000,'batch':256,'optimizer':'Adam3e-4','validation_interval':200,'selection':'Each model minimum validation own reported NLL; fixed input/data/update budget, no test tuning','models':['Unchanged native NLayerLeakyMLP,3hidden layers128; scalar inverse likelihood with native local Jacobian formula','Ordinary normalized conditional Gaussian, identical hidden sizes/depth, parameter counts differ by one'],'diagnostic':'Exact full-real-line piecewise-affine integration of phi(g)*abs(g-prime), scalar parents held fixed. Normalized density test NLL includes logZ. No global high-dimensional density certification claimed.','stop':'A native failure repaired by this existing normalized Gaussian is an implementation/objective-path issue, not a new-method result.'}
 dump(out/'plan.json',plan);verify_auditor(out)
 generator=torch.Generator(device='cuda').manual_seed(991)
 def draw(n):
  c=torch.randn(n,generator=generator,device='cuda');x=.6*c+true_sigma(c)*torch.randn(n,generator=generator,device='cuda');return c,x
 ctrain,xtrain=draw(8192);cval,xval=draw(2048);test={}
 for context in plan['test_contexts']:
  c=torch.full((8192,),context,device='cuda');test[context]=(c,.6*c+true_sigma(c)*torch.randn(8192,generator=generator,device='cuda'))
 native_cls=classes()['NLayerLeakyMLP'];rows=[]
 for seed in plan['seeds']:
  for kind in ['native_inverse','normalized_gaussian']:
   run=out/f'{kind}_s{seed}';run.mkdir(exist_ok=True);torch.manual_seed(seed)
   model=(native_cls(2,1,3,128) if kind=='native_inverse' else GaussianControl(native_cls)).cuda();optimizer=torch.optim.Adam(model.parameters(),lr=3e-4);sampler=torch.Generator(device='cuda').manual_seed(10000+seed);best=float('inf');selected=0;t0=time.perf_counter()
   def logp(c,x):return inverse_logp(model,c,x) if kind=='native_inverse' else model.logp(c,x)
   with (run/'train.jsonl').open('w',encoding='utf-8') as log:
    for step in range(1,2001):
     ids=torch.randint(0,8192,(256,),generator=sampler,device='cuda');optimizer.zero_grad(set_to_none=True);loss=-logp(ctrain[ids],xtrain[ids]).mean();assert torch.isfinite(loss);loss.backward();optimizer.step()
     if step%200==0:
      validation=-logp(cval,xval).mean().item()
      if validation<best:best=validation;selected=step;torch.save(model.state_dict(),run/'best.pt')
      log.write(json.dumps({'step':step,'train_NLL':loss.item(),'validation_reported_NLL':validation,'selected':selected})+'\n');log.flush()
   torch.save(model.state_dict(),run/'final.pt');model.load_state_dict(torch.load(run/'best.pt',map_location='cuda',weights_only=True));contexts=[]
   for context,(c,x) in test.items():
    raw=-logp(c,x).mean().item();truth=(-( -.5*((x-.6*c)/true_sigma(c)).square()-true_sigma(c).log()-.5*math.log(2*math.pi))).mean().item()
    row={'context':context,'test_reported_NLL':raw,'true_DGP_NLL':truth}
    if kind=='native_inverse':
     net=model.cpu().double();diag=audit(net,[context]);u=np.random.default_rng(seed+int(context*10)+1000).uniform(size=10000);samples=normalized_samples(net,[context],u)
     row.update(diag);row['proper_normalized_test_NLL']=raw+math.log(diag['integrated_surrogate_mass']);row['normalized_density_sample_mean']=float(samples.mean());row['normalized_density_sample_std']=float(samples.std());model=model.cuda().float()
    else:row['proper_normalized_test_NLL']=raw;row['integrated_surrogate_mass']=1.
    contexts.append(row)
   result={'kind':kind,'seed':seed,'parameters':sum(p.numel() for p in model.parameters()),'selected_step':selected,'best_validation_reported_NLL':best,'seconds':time.perf_counter()-t0,'test':contexts};rows.append(result);dump(run/'result.json',result);dump(out/'partial_results.json',rows);dump(out/'status.json',{'state':'running','completed':len(rows),'total':6});print(json.dumps(result),flush=True)
 dump(out/'results.json',rows);dump(out/'status.json',{'state':'complete','completed':6,'accepted_research_gap':False,'objective_achieved':False})
if __name__=='__main__':main()
