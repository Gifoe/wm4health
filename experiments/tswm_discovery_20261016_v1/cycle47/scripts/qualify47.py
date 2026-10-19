"""Known-law/gradient/filter qualifications; all update weights discarded."""
from common47 import *
from data47 import true_components
from scipy.special import logsumexp,gammaln
torch.set_num_threads(1)

def np_joint(model,P,rate,initial,counts,rho):
 mark=model.mark.detach().cpu().numpy();prior=model.prior.detach().cpu().numpy();grid=model.grid.detach().cpu().numpy();meanrates=None
 if model.mean_clock:
  alpha=(prior[:,None]*mark).T;alpha/=alpha.sum(-1,keepdims=True);m=[]
  for t in range(counts.shape[1]):m.append(alpha@grid);alpha=alpha@P
  with torch.no_grad():meanrates=model.rate(torch.as_tensor(np.stack(m,1),dtype=next(model.parameters()).dtype)).numpy()
 ll=[];last=[]
 for b,values,r in zip(initial,counts,rho):
  log_alpha=np.log(prior)+np.log(mark[:,b]);total_ll=logsumexp(log_alpha);log_alpha-=total_ll
  for t,c in enumerate(values):
   rates=meanrates[b,t] if model.mean_clock else rate;emission=c@np.log(mark).T+c.sum()*np.log(DT*r)-gammaln(c+1).sum()+c.sum()*np.log(rates)-DT*r*rates;log_alpha+=emission;normalizer=logsumexp(log_alpha);total_ll+=normalizer;log_alpha-=normalizer;log_alpha=logsumexp(log_alpha[:,None]+np.log(P),axis=0)
  ll.append(total_ll);last.append(np.exp(log_alpha))
 return np.array(ll),np.array(last)
def main():
 out=R/'outputs/qualification_v1';assert not out.exists();out.mkdir(parents=True)
 for src in (R/'source_archive').iterdir():
  for item in read(src/'SOURCE_TREE.json')['files']:assert sha(src/item['path'])==item['sha256']
 torch.set_default_dtype(torch.float64);seed(20264790);model=World();weights=feature_bank(8,20264791).astype('float64');grid,P,rate,mark,prior=true_components('bistable');P_t=torch.tensor(P);rate_t=torch.tensor(rate)
 # Direct independent Monte Carlo of the finite-grid Markov/Poisson model.
 rng=np.random.default_rng(20264792);all_phi=[]
 for rep in range(4):
  n=8192;state=rng.choice(K,size=n,p=prior);initial=(rng.random(n)[:,None]>np.cumsum(mark[state],-1)).sum(-1);counts=np.empty((n,T,8),dtype='int16')
  for t in range(T):counts[:,t]=rng.poisson(DT*rate[state,None]*mark[state]);state=(rng.random(n)[:,None]>np.cumsum(P[state],-1)).sum(-1)
  all_phi.append(empirical_features(initial,counts,weights))
 phi=np.concatenate(all_phi);exact=model.laplace(torch.tensor(weights),1.,P_t,rate_t).detach().numpy();se=phi.std(0,ddof=1)/math.sqrt(len(phi));zs=np.abs(phi.mean(0)-exact)/se;assert max(zs)<5.5,zs;np.savez_compressed(out/'laplace_mc.npz',features=phi,exact=exact,weights=weights)
 # Differentiable neural parameters versus central finite differences.
 Pn,rn=model.components();loss=model.laplace(torch.tensor(weights),1.,Pn,rn).mean();loss.backward();gradient_checks=[]
 for name,param in [('decay',model.raw_decay),('diffusion',model.raw_sigma),('clock_bias',model.rnet._model[-1].bias),('drift_bias',model.fnet._model[-1].bias)]:
  actual=float(param.grad.reshape(-1)[0]);original=float(param.detach().reshape(-1)[0]);h=1e-5;values=[]
  for shift in [h,-h]:
   with torch.no_grad():param.reshape(-1)[0]=original+shift;values.append(float(model.laplace(torch.tensor(weights),1.).mean()))
  with torch.no_grad():param.reshape(-1)[0]=original
  expected=(values[0]-values[1])/(2*h);error=abs(actual-expected);assert error<3e-7,(name,actual,expected);gradient_checks.append({'parameter':name,'autograd':actual,'finite_difference':expected,'absolute_error':error})
 train=np.load(R/'data/ou/train.npz');ini=train['initial'][:4];cts=train['counts'][:4,:6].astype('float64');rho=train['rho'][:4].astype('float64');filter_errors=[]
 for mean in [False,True]:
  model.mean_clock=mean;Pn,rn=model.components();ll,post=model.joint(torch.tensor(ini),torch.tensor(cts),torch.tensor(rho),Pn,rn,True);ell,ep=np_joint(model,Pn.detach().numpy(),rn.detach().numpy(),ini,cts,rho);error={'mean_clock':mean,'log_likelihood':float(np.max(abs(ll.detach().numpy()-ell))),'posterior':float(np.max(abs(post.detach().numpy()-ep)))};assert max(error['log_likelihood'],error['posterior'])<1e-8,error;filter_errors.append(error)
 # Future begins at the filtered boundary state, with no extra transition.
 model.mean_clock=False;initial=torch.tensor(train['initial'][:1]);counts=torch.tensor(train['counts'][:1,:HIST],dtype=torch.float64);r=torch.tensor(train['rho'][:1]);Pn,rn=model.components();_,q=model.joint(initial,counts,r,Pn,rn,True);sample=model.sample(initial,counts,r,16384,20264793);alpha=q.detach().numpy()[0];expected_mean=[];expected_var=[]
 for t in range(FUTURE):expected_mean.append(alpha@grid);expected_var.append(alpha@(grid**2)-(alpha@grid)**2+OBS**2);alpha=alpha@Pn.detach().numpy()
 observed=sample[0].numpy();mean_z=np.max(abs(observed.mean(0)-expected_mean)/np.sqrt(np.asarray(expected_var)/len(observed)));assert mean_z<5.5,mean_z;np.savez_compressed(out/'boundary_future.npz',draws=observed,exact_mean=expected_mean,exact_variance=expected_var)
 torch.set_default_dtype(torch.float32);bank=np.load(R/'data/feature_bank.npz')['weights'];stats=np.load(R/'data/ou/train_features.npz');timings=[]
 for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
  for method in METHODS:
   seed(20264794);w=World(method=='mean_nll').to(device);opt=torch.optim.Adam(w.parameters(),lr=1e-3);initial=torch.as_tensor(train['initial'][:32],device=device);counts=torch.as_tensor(train['counts'][:32],dtype=torch.float32,device=device);rho=torch.as_tensor(train['rho'][:32],device=device);ww=torch.as_tensor(bank[:64],device=device);mu=torch.as_tensor(stats['mean'][0,:64],device=device);var=torch.as_tensor(stats['variance'][0,:64],device=device);dur=[];losses=[]
   for step in range(7):
    opt.zero_grad(set_to_none=True)
    if device=='cuda':torch.cuda.synchronize()
    start=time.perf_counter();loss,_=objective(w,method,initial,counts,rho,ww,mu,var,.5);assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(w.parameters(),10.);assert torch.isfinite(norm);opt.step()
    if device=='cuda':torch.cuda.synchronize()
    dur.append(time.perf_counter()-start);losses.append(float(loss.detach().cpu()))
   timings.append({'device':device,'method':method,'discarded_updates':7,'median_step_seconds':float(np.median(dur[2:])),'parameters':sum(p.numel() for p in w.parameters()),'losses':losses})
 device=min(sorted(set(x['device'] for x in timings)),key=lambda d:np.mean([x['median_step_seconds'] for x in timings if x['device']==d]));result={'status':'QUALIFIED_FOR_FROZEN_FINITE_GRID_SCREEN','utc':now(),'laplace_MC_paths':len(phi),'max_Laplace_MC_SE':float(max(zs)),'gradient_checks':gradient_checks,'independent_filter_errors':filter_errors,'max_future_boundary_mean_MC_SE':float(mean_z),'selected_training_device':device,'timings':timings,'source_scope':'unchanged selected MLP_standard class definitions only; not full original pipeline','source_component_sha256':sha(component_path),'common_sha256':sha(R/'scripts/common47.py'),'qualification_source_sha256':sha(__file__),'all_updates_discarded':True,'EVAL_opened':False};dump(out/'qualification.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
