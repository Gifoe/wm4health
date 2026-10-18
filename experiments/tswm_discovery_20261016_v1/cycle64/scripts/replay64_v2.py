"""Independent saved-artifact replay; no fitting choices or test selection."""
from common64 import *
import numpy as np
import scipy.linalg as sla
from scipy.special import ndtr
from collections import defaultdict,Counter
f=read(OUT/'FIT_AND_GAUSSIAN_EVALUATION64_V1.json');s=read(OUT/'SAMPLER_EVALUATION64_V1.json');c=read(OUT/'CAUSAL_FACTORIZATION64_V1.json')
source_check();errors=defaultdict(float);counts=Counter();invalid=[]
def check(label,a,b,tol=2e-7):
 a=np.asarray(a);b=np.asarray(b);err=float(np.max(np.abs(a-b)/(1+np.abs(b))));errors[label]=max(errors[label],err);assert err<=tol,(label,err)
def truth(u):
 h=len(u);F=np.zeros((h,h))
 for i in range(h):
  for j in range(i+1):F[i,j]=.8**(i-j)
 return .8**(np.arange(h)+1)*.25+F@(.7*u),.25*F@F.T
def met(m,C,tm,tC):
 # Independent Bures expression via singular values of covariance factors.
 R=sla.cholesky(tC,lower=True);V=sla.cholesky(C+1e-14*np.eye(len(C)),lower=True)
 w=max(0.,float((m-tm)@(m-tm)+np.trace(C)+np.trace(tC)-2*sla.svdvals(R.T@V).sum()))
 tail=float(ndtr((-2-m[-1])/np.sqrt(C[-1,-1]))+ndtr((m[-1]-2)/np.sqrt(C[-1,-1])))
 tt=float(ndtr((-2-tm[-1])/np.sqrt(tC[-1,-1]))+ndtr((tm[-1]-2)/np.sqrt(tC[-1,-1])))
 return dict(gaussian_W2_squared=w,normalized_Gaussian_moment_discrepancy=np.sqrt(w/np.trace(tC)),mean_MSE=np.mean((m-tm)**2),variance_trace_ratio=np.trace(C)/np.trace(tC),terminal_tail_probability=tail,true_terminal_tail_probability=tt,terminal_tail_abs_error=abs(tail-tt))
def verify_metrics(row,m,C,tm,tC,particle=False,x=None,w=None):
 out=met(m,C,tm,tC)
 if particle:
  out['Gaussian_moment_projection_W2_squared']=out.pop('gaussian_W2_squared');out['Gaussian_moment_tail_probability']=out['terminal_tail_probability'];out['terminal_tail_probability']=float(w@(np.abs(x[:,-1])>2));out['terminal_tail_abs_error']=abs(out['terminal_tail_probability']-out['true_terminal_tail_probability'])
 for key,val in out.items():check('metric_'+key,row[key],val,tol=5e-7)
def query(h,pk):return np.zeros(h) if pk=='zero' else .4*np.sin(2*np.pi*(np.arange(h)+1)/8)
def gauss_nll(res,C):
 sg,ld=np.linalg.slogdet(C);assert sg>0
 return float(.5*(len(C)*np.log(2*np.pi)+ld+np.einsum('ni,ij,nj->n',res,np.linalg.inv(C),res).mean())/len(C))
data={}
for r in f['corpora']:
 assert sha(r['path'])==r['sha256'];z=np.load(r['path']);data[r['key']]={k:z[k] for k in z.files};counts['data_hashes']+=1
factor_data=np.load(c['artifact']);assert sha(c['artifact'])==c['artifact_sha256']
for fit in f['fits']:
 assert sha(fit['path'])==fit['sha256'];z=np.load(fit['path']);counts['checkpoint_hashes']+=1;h=fit['horizon'];k=fit['gain'];d=data[fit['key']];U=d['U'][:,:h];Y=d['Y'][:,:h];N=len(U);Um=z['joint_mean'][h:];Ym=z['joint_mean'][:h];Cj=z['joint_covariance']
 # Alternate QR regression and covariance from residuals, versus sample Schur.
 design=np.c_[np.ones(N),U];coef=sla.lstsq(design,Y,lapack_driver='gelsy')[0];res=Y-design@coef;condcov=res.T@res/N;B=coef[1:].T;inter=coef[0]
 fittedC=Cj[:h,:h]-Cj[:h,h:]@np.linalg.solve(Cj[h:,h:],Cj[h:,:h]);check('conditional_covariance_QR',condcov,fittedC,3e-9)
 Uv=d['U_validation'][:,:h];Yv=d['Y_validation'][:,:h];check('observational_validation_NLL',gauss_nll(Yv-(inter+Uv@B.T),condcov),fit['validation_observed_conditional_NLL_per_step']['block_Gaussian'],3e-8)
 prev=np.c_[np.full(N,.25),Y[:,:-1]].reshape(-1);X=np.c_[prev,U.reshape(-1),np.ones(N*h)];ar=sla.lstsq(X,Y.reshape(-1),lapack_driver='gelsy')[0];r=Y.reshape(-1)-X@ar;nv=np.mean(r*r);check('AR_QR_predictions',X@ar,X@z['ar']);check('AR_QR_variance',nv,z['noise_variance'])
 Xp=np.c_[prev,np.ones(N*h)];pol=sla.lstsq(Xp,U.reshape(-1),lapack_driver='gelsy')[0];pr=U.reshape(-1)-Xp@pol;pv=np.mean(pr*pr);check('policy_QR_predictions',Xp@pol,Xp@z['policy']);check('policy_QR_variance',pv,z['policy_variance'])
 # Direct regressions of each Y_t against its predecessors in the frozen JOINT.
 order=np.array([q for t in range(h) for q in (h+t,t)]);mu=z['joint_mean'][order];C=Cj[np.ix_(order,order)];A=np.eye(h);D=np.zeros((h,h));b=np.zeros(h);v=np.empty(h)
 for t in range(h):
  j=2*t+1;cc=np.linalg.solve(C[:j,:j],C[:j,j]);A[t,:t]=-cc[1::2];D[t,:t+1]=cc[::2];b[t]=mu[j]-cc@mu[:j];v[t]=C[j,j]-cc@C[:j,j]
 key=fit['key']+'_H'+str(h)
 for n,a in [('A',A),('B',D),('b',b),('variance',v)]:check('joint_causalization_'+n,a,factor_data[key+'_'+n])
 Ai=np.linalg.inv(A);Cfactor=(Ai*v)@Ai.T
 for pk in ['zero','sine']:
  u=query(h,pk);tm,tC=truth(u);mo=inter+B@u;Qo=np.linalg.inv(fittedC);ho=Qo@(Ym+Cj[:h,h:]@np.linalg.solve(Cj[h:,h:],u-Um))
  L=np.eye(h)-z['ar'][0]*np.eye(h,k=-1);bb=z['ar'][1]*u+z['ar'][2];bb[0]+=z['ar'][0]*.25;Li=np.linalg.inv(L);ma=Li@bb;Ca=float(z['noise_variance'])*Li@Li.T
  laws={'observational_block':(mo,fittedC),'causal_AR':(ma,Ca)}
  for label,pc,pvv,pd in [('known_policy',-k,.25,0.),('estimated_policy',z['policy'][0],float(z['policy_variance']),z['policy'][1])]:
   Q=Qo-np.diag(np.r_[np.repeat(pc*pc/pvv,h-1),0.]);hh=ho-np.r_[pc*(u[1:]-pd)/pvv,0.];ev,V=np.linalg.eigh(Q)
   laws['ratio_'+label]=None if ev[0]<=0 else (np.linalg.solve(Q,hh),np.linalg.inv(Q))
   if label=='estimated_policy':
    Qf=(V*np.maximum(ev,1e-3))@V.T;laws['fixed_eigenvalue_floor']=(np.linalg.solve(Qf,hh),np.linalg.inv(Qf))
  rows=[r for r in f['evaluation'] if r['gain']==k and r['data_seed']==fit['data_seed'] and r['horizon']==h and r['query']==pk]
  for r in rows:
   law=laws[r['method']];assert r['valid']==(law is not None)
   if law is None:invalid.append({q:r[q] for q in ['gain','horizon','data_seed','query','method']})
   else:verify_metrics(r,*law,tm,tC)
   counts['learned_rows']+=1
  rr=next(r for r in c['rows'] if r['gain']==k and r['data_seed']==fit['data_seed'] and r['horizon']==h and r['query']==pk);verify_metrics(rr,Ai@(b+D@u),Cfactor,tm,tC);counts['same_joint_rows']+=1
assert sha(s['artifact'])==s['artifact_sha256'];raw=np.load(s['artifact'])
for row in s['particle_rows']:
 h=row['horizon'];k=row['gain'];P=row['particles'];seed=row['MC_seed'];prefix='IS' if row['method']=='terminal_IS' else 'SMC';key=f'{prefix}_k{k}_H{h}_P{P}_s{seed}';x=raw[key+'_x'];w=raw[key+'_w'];check('weight_normalization',w.sum(),1.);assert (w>=0).all();m=w@x;r=x-m;C=np.einsum('n,ni,nj->ij',w,r,r);tm,tC=truth(query(h,'sine'));verify_metrics(row,m,C,tm,tC,True,x,w);check('empirical_ESS',row['empirical_ESS'],1/(w@w));counts['particle_rows']+=1
for row in s['DDIM_law_rows']:
 h=row['horizon'];key=f'DDIM_{row["method"]}_k{row["gain"]}_H{h}_steps{row["DDIM_steps"]}';tm,tC=truth(query(h,'sine'));verify_metrics(row,raw[key+'_mean'],raw[key+'_cov'],tm,tC);counts['DDIM_moment_rows']+=1
assert counts==dict(data_hashes=12,checkpoint_hashes=48,learned_rows=480,same_joint_rows=96,particle_rows=168,DDIM_moment_rows=144),counts
dump(OUT/'INDEPENDENT_REPLAY64_V1.json',dict(utc=utc(),status='PASS',counts=dict(counts),max_scaled_errors=dict(errors),invalid_rows=invalid,scope='Independent SciPy QR refits, structural truth, direct sequential Gaussian regressions, SVD Bures moments, all saved weights and DDIM final-law metrics. Does not independently execute a neural diffusion model or prove novelty.'))
groups=[]
for H in [1,4,16,64]:
 for k in [0.,.25,.75,2.]:
  for method in ['observational_block','causal_AR','ratio_known_policy','ratio_estimated_policy','fixed_eigenvalue_floor','same_frozen_joint_temporal_causal_factorization']:
   rr=[r for r in f['evaluation']+c['rows'] if r['horizon']==H and r['gain']==k and r['method']==method];valid=[r for r in rr if r['valid']]
   groups.append(dict(gain=k,horizon=H,method=method,rows=len(rr),invalid=len(rr)-len(valid),mean_discrepancy=float(np.mean([r['normalized_Gaussian_moment_discrepancy'] for r in valid])) if valid else None))
smc=[]
for k in [.75,2.]:
 for H in [16,64]:
  for P in [128,512]:
   rr=[r for r in s['particle_rows'] if r['method']!='terminal_IS' and r['gain']==k and r['horizon']==H and r['particles']==P];smc.append(dict(gain=k,horizon=H,particles=P,passed=sum(r['operational_pass'] for r in rr),replications=len(rr),mean_discrepancy=float(np.mean([r['normalized_Gaussian_moment_discrepancy'] for r in rr])),max_tail_error=max(r['terminal_tail_abs_error'] for r in rr)))
dd=[]
for k in [.75,2.]:
 for H in [16,64]:
  for method in ['observed','plugin','exact_target']:
   r=next(r for r in s['DDIM_law_rows'] if r['gain']==k and r['horizon']==H and r['method']==method and r['DDIM_steps']==1024);dd.append(r)
dump(OUT/'SUMMARY64_V1.json',dict(utc=utc(),status='CLOSED_GAUSSIAN_ATTRIBUTION_NOT_ACCEPTED_DIRECTION',accepted_final_directions=0,learned_groups=groups,invalid=invalid,validation_nll=[{k:r[k] for k in ['gain','data_seed','horizon','validation_observed_conditional_NLL_per_step']} for r in f['fits']],population_IS=s['population_IS'],SMC=smc,DDIM_1024=dd,score_examples=[r for r in s['noisy_score_rows'] if r['gain']==2. and r['horizon']==64],fit_runtime_seconds=f['elapsed_seconds'],sampler_runtime_seconds=s['runtime_seconds'],replay_sha256=sha(OUT/'INDEPENDENT_REPLAY64_V1.json')))
print(json.dumps({'status':'PASS','counts':dict(counts),'max_scaled_error':max(errors.values()),'SMC':smc,'H64_gain2_groups':[r for r in groups if r['horizon']==64 and r['gain']==2.],'DDIM_1024':dd,'invalid_by_method':dict(Counter(r['method'] for r in invalid))}))
