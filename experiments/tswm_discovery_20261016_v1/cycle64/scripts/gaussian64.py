from common64 import *
import numpy as np
import scipy.linalg as la
from scipy.special import logsumexp,ndtr
RHO=.8;BETA=.7;SIGMA=.5;TAU=.5;Y0=.25
GAINS=[0.,.25,.75,2.];HORIZONS=[1,4,16,64];SEEDS=[2021,2022,2023]
def sym(a):return (a+a.T)/2
def inverse(a):return la.cho_solve(la.cho_factor(sym(a),lower=True),np.eye(len(a)))
def plan(h,kind):return np.zeros(h) if kind=='zero' else .4*np.sin(2*np.pi*(np.arange(h)+1)/8)
def canonical(u,c,rho=RHO,beta=BETA,sigma=SIGMA,tau=TAU,intercept=0.,policy_intercept=0.):
 h=len(u);L=np.eye(h)-rho*np.eye(h,k=-1);b=beta*np.asarray(u)+intercept;b=b.copy();b[0]+=rho*Y0;Q=L.T@L/sigma**2;hd=L.T@b/sigma**2;md=la.solve_triangular(L,b,lower=True);Cd=inverse(Q)
 p=np.r_[np.repeat(c*c/tau**2,h-1),0.];ell=np.r_[c*(np.asarray(u)[1:]-policy_intercept)/tau**2,0.];Qo=Q+np.diag(p);ho=hd+ell;Co=inverse(Qo);mo=Co@ho
 return {'do_mean':md,'do_cov':Cd,'do_Q':Q,'do_h':hd,'obs_mean':mo,'obs_cov':Co,'obs_Q':Qo,'obs_h':ho,'p_diag':p,'ell':ell}
def structural_joint(h,c,rho=RHO,beta=BETA,sigma=SIGMA,tau=TAU,intercept=0.,policy_intercept=0.):
 a=rho+beta*c;g=np.zeros((h,h))
 for t in range(h):g[t,:t+1]=a**np.arange(t,-1,-1)
 ym=[];last=Y0
 for t in range(h):last=a*last+beta*policy_intercept+intercept;ym.append(last)
 ym=np.array(ym);um=c*np.r_[Y0,ym[:-1]]+policy_intercept;Ym=np.concatenate([sigma*g,beta*tau*g],1);shift=np.eye(h,k=-1);Um=c*shift@Ym+np.concatenate([np.zeros((h,h)),tau*np.eye(h)],1);M=np.concatenate([Ym,Um],0)
 return np.r_[ym,um],M@M.T
def conditional(jm,jc,u):
 h=len(u);S=jc[:h,h:];B=la.solve(jc[h:,h:],S.T,assume_a='pos').T;C=sym(jc[:h,:h]-B@S.T);m=jm[:h]+B@(np.asarray(u)-jm[h:]);return m,C,B
def moments_from_precision(Q,h):
 Q=sym(Q);e,V=la.eigh(Q);minimum=float(e[0])
 if minimum<=0:return {'valid':False,'minimum_precision_eigenvalue':minimum}
 C=(V/e)@V.T;m=C@h;return {'valid':True,'minimum_precision_eigenvalue':minimum,'mean':m,'cov':sym(C)}
def metrics(m,C,target_m,target_C):
 e,V=la.eigh(sym(target_C));root=(V*np.sqrt(np.maximum(e,0)))@V.T;ev=la.eigvalsh(sym(root@C@root));w2=float(np.sum((m-target_m)**2)+np.trace(C)+np.trace(target_C)-2*np.sqrt(np.maximum(ev,0)).sum());w2=max(w2,0.)
 threshold=2.;sd=float(np.sqrt(max(C[-1,-1],1e-20)));tsd=float(np.sqrt(target_C[-1,-1]));tail=float(ndtr((-threshold-m[-1])/sd)+ndtr((m[-1]-threshold)/sd));true_tail=float(ndtr((-threshold-target_m[-1])/tsd)+ndtr((target_m[-1]-threshold)/tsd))
 return {'gaussian_W2_squared':w2,'normalized_Gaussian_moment_discrepancy':float(np.sqrt(w2/np.trace(target_C))),'mean_MSE':float(np.mean((m-target_m)**2)),'variance_trace_ratio':float(np.trace(C)/np.trace(target_C)),'terminal_tail_probability':tail,'true_terminal_tail_probability':true_tail,'terminal_tail_abs_error':abs(tail-true_tail)}
def logpdf(x,m,C):
 L=la.cholesky(C,lower=True);v=la.solve_triangular(L,(x-m).T,lower=True);return -.5*(len(m)*np.log(2*np.pi)+2*np.log(np.diag(L)).sum()+np.sum(v*v,axis=0))
def population_IS(g):
 Qt=g['do_Q'];Qo=g['obs_Q'];mt=g['do_mean'];mo=g['obs_mean'];A=2*Qt-Qo;e=la.eigvalsh(sym(A));minimum=float(e[0])
 if minimum<=0:return {'finite_second_moment':False,'minimum_second_moment_precision_eigenvalue':minimum,'log_second_moment':None,'population_ESS_fraction':0.}
 hh=2*Qt@mt-Qo@mo;logI=float(np.linalg.slogdet(Qt)[1]-.5*np.linalg.slogdet(Qo)[1]-.5*np.linalg.slogdet(A)[1]-mt@Qt@mt+.5*mo@Qo@mo+.5*hh@la.solve(A,hh,assume_a='pos'));assert logI>=-1e-8
 return {'finite_second_moment':True,'minimum_second_moment_precision_eigenvalue':minimum,'log_second_moment':max(logI,0.),'population_ESS_fraction':float(np.exp(-max(logI,0.)))}
def score_affine(m,C,a):
 N=a*C+(1-a)*np.eye(len(C));Q=inverse(N);return -Q,Q@(np.sqrt(a)*m)
def plugin_score(g,a):
 mo,Co=g['obs_mean'],g['obs_cov'];S,b=score_affine(mo,Co,a);N=a*Co+(1-a)*np.eye(len(Co));J=np.sqrt(a)*Co@inverse(N);v=mo-J@(np.sqrt(a)*mo);P=np.diag(g['p_diag']);return S+J.T@P@J,b+J.T@(P@v-g['ell'])
def score_error(g,a):
 S,b=plugin_score(g,a);St,bt=score_affine(g['do_mean'],g['do_cov'],a);C=a*g['do_cov']+(1-a)*np.eye(len(S));m=np.sqrt(a)*g['do_mean'];D=S-St;v=D@m+b-bt;return float((np.trace(D@C@D.T)+v@v)/np.trace(inverse(C)))
