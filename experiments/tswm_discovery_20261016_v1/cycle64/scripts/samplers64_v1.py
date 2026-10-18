from gaussian64 import *
from native_smc64 import load
import torch,time
assert read(OUT/'QUALIFICATION64_V1.json')['status']=='PASS';fk,util,compat=load();torch.set_num_threads(4);pp=paths();rows=[];pop=[];score_rows=[];ddim=[];raw={};t0=time.perf_counter()
def particle_metrics(x,w,g):
 m=np.sum(x*w[:,None],axis=0);r=x-m;C=sym((r*w[:,None]).T@r);out=metrics(m,C,g['do_mean'],g['do_cov']);out['Gaussian_moment_projection_W2_squared']=out.pop('gaussian_W2_squared');out['Gaussian_moment_tail_probability']=out['terminal_tail_probability'];tail=float(np.sum(w*(np.abs(x[:,-1])>2.)));out['terminal_tail_probability']=tail;out['terminal_tail_abs_error']=abs(tail-out['true_terminal_tail_probability']);out['metric_scope']='Weighted empirical law: Gaussian moment-projection discrepancy and actual weighted empirical tail, not exact empirical-law W2';return out
def ddim_law(g,method,steps):
 h=len(g['do_mean']);m=np.zeros(h);C=np.eye(h);start=1e-5;end=1-1e-5;l=np.linspace(np.log(start/(1-start)),np.log(end/(1-end)),steps);aa=np.r_[1/(1+np.exp(-l)),1.];I=np.eye(h)
 for a,ap in zip(aa[:-1],aa[1:]):
  if method=='plugin':S,b=plugin_score(g,a)
  else:
   prefix='do' if method=='exact_target' else 'obs';S,b=score_affine(g[prefix+'_mean'],g[prefix+'_cov'],a)
  r=np.sqrt(ap/a);factor=r*(1-a)-np.sqrt((1-ap)*(1-a));M=r*I+factor*S;d=factor*b;m=M@m+d;C=sym(M@C@M.T)
 return m,C
for k in GAINS:
 for h in HORIZONS:
  state('sampler_state64_v1.json',{'status':'IS_AND_SCORE_CONTROLS','gain':k,'horizon':h,'completed_rows':len(rows)})
  g=canonical(plan(h,'sine'),-k);p=population_IS(g);pop.append({'gain':k,'horizon':h,**p});chol=la.cholesky(g['obs_cov'],lower=True)
  for P in [128,512,2048]:
   for s in SEEDS:
    rng=np.random.default_rng(np.random.SeedSequence([64100,s,int(k*100),h,P]));x=rng.normal(size=(P,h))@chol.T+g['obs_mean'];lw=logpdf(x,g['do_mean'],g['do_cov'])-logpdf(x,g['obs_mean'],g['obs_cov']);w=np.exp(lw-logsumexp(lw));key=f'IS_k{k}_H{h}_P{P}_s{s}';raw[key+'_x']=x;raw[key+'_w']=w;rows.append({'method':'terminal_IS','gain':k,'horizon':h,'particles':P,'MC_seed':s,'empirical_ESS':float(1/np.sum(w*w)),'population_ESS_fraction':p['population_ESS_fraction'],**particle_metrics(x,w,g)})
  for a in [.01,.1,.5,.9,.99]:score_rows.append({'gain':k,'horizon':h,'alpha':a,'plugin_relative_expected_score_MSE':score_error(g,a)})
  for steps in [64,256,1024]:
   for method in ['observed','plugin','exact_target']:
    m,C=ddim_law(g,method,steps);assert np.isfinite(m).all() and np.isfinite(C).all();key=f'DDIM_{method}_k{k}_H{h}_steps{steps}';raw[key+'_mean']=m;raw[key+'_cov']=C;ddim.append({'method':method,'gain':k,'horizon':h,'DDIM_steps':steps,'initial_alpha':1e-5,'initial_law':'N(0,I), shared; residual alpha initialization error reported through exact-target control',**metrics(m,C,g['do_mean'],g['do_cov'])})
  print(json.dumps({'stage':'IS_DDIM','gain':k,'horizon':h,'finite_IS_second_moment':p['finite_second_moment']}),flush=True)
for k in [.75,2.]:
 for h in [16,64]:
  g=canonical(plan(h,'sine'),-k);pdiag=torch.tensor(g['p_diag'],dtype=torch.float64);ell=torch.tensor(g['ell'],dtype=torch.float64);Qo=torch.tensor(g['obs_Q'],dtype=torch.float64);ho=torch.tensor(g['obs_h'],dtype=torch.float64);mo=torch.tensor(g['obs_mean'],dtype=torch.float64);Lo=torch.tensor(la.cholesky(g['obs_cov'],lower=True),dtype=torch.float64);J=32
  # Exact tridiagonal Gibbs conditional uses only the available Gaussian energy.
  off=Qo.clone()
  for i in range(h):off[i,max(0,i-1):min(h,i+2)]=0
  assert float(off.abs().max())<1e-12
  for P in [128,512]:
   for s in SEEDS:
    state('sampler_state64_v1.json',{'status':'NATIVE_FK_ANNEALED_GIBBS','gain':k,'horizon':h,'particles':P,'seed':s});np.random.seed(s);torch.manual_seed(s);starttime=time.perf_counter()
    def M(t,xtp1,extra_vals,P):
     if t==J:return torch.randn((P,h),dtype=torch.float64)@Lo.T+mo,extra_vals
     previous=(J-t-1)/J;Q=Qo-torch.diag(previous*pdiag);hh=ho-previous*ell;x=xtp1.clone()
     for i in range(h):
      numerator=hh[i].expand(P).clone()
      if i>0:numerator-=Q[i,i-1]*x[:,i-1]
      if i+1<h:numerator-=Q[i,i+1]*x[:,i+1]
      x[:,i]=numerator/Q[i,i]+torch.randn(P,dtype=torch.float64)/torch.sqrt(Q[i,i])
     return x,extra_vals
    def G(t,xtp1,xt,extra_vals):
     if t==J:return torch.zeros(P,dtype=torch.float64)
     return (.5*(xt.square()*pdiag).sum(-1)-xt@ell)/J
    x,lw,w,indices,ess,lwt,_=fk(M,G,'systematic',J,P,.5,log_xt_trace=False);x=x.numpy();w=w.numpy();key=f'SMC_k{k}_H{h}_P{P}_s{s}';raw[key+'_x']=x;raw[key+'_w']=w;raw[key+'_ess']=ess.numpy();raw[key+'_logw_trace']=lwt.numpy();row={'method':'ordinary_annealed_Gaussian_Gibbs_SMC_native_FK','gain':k,'horizon':h,'particles':P,'MC_seed':s,'temperatures':J,'Gibbs_sweeps_per_temperature':1,'runtime_seconds':time.perf_counter()-starttime,'empirical_ESS':float(1/np.sum(w*w)),**particle_metrics(x,w,g)};row['operational_pass']=row['normalized_Gaussian_moment_discrepancy']<=.1 and row['terminal_tail_abs_error']<=.03;rows.append(row)
   print(json.dumps({'stage':'SMC','gain':k,'horizon':h,'particles':P}),flush=True)
artifact=pp['artifact_root']/'SAMPLER_TRACES64_V1.npz';np.savez_compressed(artifact,**raw)
dump(OUT/'SAMPLER_EVALUATION64_V1.json',{'utc':utc(),'status':'ALL_FIXED_SAMPLER_CONTROLS_SCORED','population_IS':pop,'particle_rows':rows,'noisy_score_rows':score_rows,'DDIM_law_rows':ddim,'artifact':str(artifact),'artifact_sha256':sha(artifact),'runtime_seconds':time.perf_counter()-t0,'native_generic_FK_compatibility':compat,'scope':'Oracle Gaussian sampler attribution, existing Gaussian Gibbs annealing and native FK utility; not a learned diffusion-model experiment or full TDS reproduction.','accepted_final_directions':0})
state('sampler_state64_v1.json',{'status':'ALL_FIXED_SAMPLER_CONTROLS_CLOSED','particle_rows':len(rows),'DDIM_rows':len(ddim),'score_rows':len(score_rows)});print(json.dumps({'status':'ALL_FIXED_SAMPLER_CONTROLS_CLOSED','particle_rows':len(rows),'DDIM_rows':len(ddim)}))
