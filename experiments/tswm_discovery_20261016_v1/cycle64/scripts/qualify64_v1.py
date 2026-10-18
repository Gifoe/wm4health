from gaussian64 import *
from native_smc64 import load
import torch
errors=[]
for k in GAINS:
 for h in HORIZONS:
  u=plan(h,'sine');g=canonical(u,-k);jm,jc=structural_joint(h,-k);m,C,_=conditional(jm,jc,u);err=max(float(np.max(np.abs(m-g['obs_mean']))),float(np.max(np.abs(C-g['obs_cov']))));assert err<1e-9,(k,h,err);errors.append(err)
  assert np.max(np.abs(g['obs_Q']-g['do_Q']-np.diag(g['p_diag'])))<1e-12
  if k==0 or h==1:assert np.max(np.abs(g['obs_mean']-g['do_mean']))<1e-12 and np.max(np.abs(g['obs_cov']-g['do_cov']))<1e-12
g=canonical(plan(2,'sine'),-2.);a=.4;z=np.array([.2,-.7]);P=np.diag(g['p_diag']);R=g['obs_Q']+a/(1-a)*np.eye(2);S=R-P
def log_e(v):
 hh=g['obs_h']+np.sqrt(a)/(1-a)*v;return .5*np.linalg.slogdet(R)[1]-.5*np.linalg.slogdet(S)[1]+.5*(hh-g['ell'])@la.solve(S,hh-g['ell'],assume_a='pos')-.5*hh@la.solve(R,hh,assume_a='pos')
fd=np.array([(log_e(z+np.eye(2)[j]*1e-5)-log_e(z-np.eye(2)[j]*1e-5))/2e-5 for j in range(2)]);St,bt=score_affine(g['do_mean'],g['do_cov'],a);So,bo=score_affine(g['obs_mean'],g['obs_cov'],a);score_err=float(np.max(np.abs(fd-((St-So)@z+bt-bo))));assert score_err<1e-6
fk,util,compat=load();np.random.seed(6401);torch.manual_seed(6401);torch.set_num_threads(4)
w=torch.zeros(8,dtype=torch.float64);idx,res=util.resampling_function('systematic',None)(w);assert res and torch.equal(idx,torch.arange(8))
ess=float(util.compute_ess_from_log_w(torch.log(torch.tensor([1.,2.,3.],dtype=torch.float64))));assert abs(ess-36/14)<1e-12
def M(t,xtp1,extra_vals,P):return torch.randn(P,dtype=torch.float64),extra_vals
def G(t,xtp1,xt,extra_vals):return .3*xt
x,lw,nw,*_=fk(M,G,'systematic',0,16384,.5);tm=float((x*nw).sum());assert abs(tm-.3)<.04
dump(OUT/'QUALIFICATION64_V1.json',{'utc':utc(),'status':'PASS','max_oracle_conditional_difference':max(errors),'oracle_cases':len(errors),'noisy_log_ratio_score_fd_error':score_err,'native_smc_weight_ESS':ess,'native_smc_tilt_fixture_mean':tm,'native_smc_tilt_truth_mean':.3,'compatibility':compat,'source_receipt_sha256':sha(OUT/'SOURCES64_V1.json'),'protocol_sha256':sha(ROOT/'PROTOCOL64_V1.md'),'scope':'Generic author Feynman-Kac utility, not full TDS denoiser reproduction. Model/algorithm functions unchanged except optional JIT decorator removal.'})
print(json.dumps({'status':'PASS','oracle_error':max(errors),'score_fd_error':score_err,'native_smc_mean':tm}))
