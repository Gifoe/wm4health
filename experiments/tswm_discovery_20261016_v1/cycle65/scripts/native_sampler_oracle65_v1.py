from models65 import *
import scipy.linalg as sla
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest');torch.set_num_threads(4)
registry=OUT/'FITS65_V1.json' if (OUT/'FITS65_V1.json').exists() else OUT/'fit_registry65_v1.json';f=read(registry);device=read(OUT/'QUALIFICATION65_V1.json')['device'];sampler=csdi(device);rows=[];checks=[];art={};N=8192
aa=.8-.7*.75;G=np.zeros((H,H))
for t in range(H):G[t,:t+1]=aa**np.arange(t,-1,-1)
ym=aa**(np.arange(H)+1)*.25;um=-.75*np.r_[.25,ym[:-1]];MY=np.c_[.5*G,.7*.5*G];MU=-.75*np.eye(H,k=-1)@MY+np.c_[np.zeros((H,H)),.5*np.eye(H)];MM=np.r_[MY,MU];mu=np.r_[ym,um];C=MM@MM.T
def conditional(mu,C,O):
 U=np.setdiff1d(np.arange(2*H),O);B=np.linalg.solve(C[np.ix_(O,O)],C[np.ix_(O,U)]).T;cc=C[np.ix_(U,U)]-B@C[np.ix_(O,U)];return U,B,(cc+cc.T)/2
def propagate(cc):
 ev,V=np.linalg.eigh(cc);coef=np.zeros(len(ev));var=np.ones(len(ev))
 for t in range(49,-1,-1):
  a=sampler.alpha[t];ah=sampler.alpha_hat[t];beta=sampler.beta[t];nv=a*ev+1-a;A=(1-beta/nv)/np.sqrt(ah);b=beta*np.sqrt(a)/(np.sqrt(ah)*nv);vv=((1-sampler.alpha[t-1])/(1-a)*beta) if t>0 else 0.;coef=A*coef+b;var=A*A*var+vv
 return (V*coef)@V.T,(V*var)@V.T
def law(mean,std,u,operation):
 sd=np.repeat(std.reshape(-1),H);mm=np.repeat(mean.reshape(-1),H);mn=(mu-mm)/sd;cn=C/np.outer(sd,sd)
 if operation=='block':
  O=np.arange(H,2*H);U,B,cc=conditional(mn,cn,O);F,V=propagate(cc);m=F@(mn[U]+B@((u-mm[O])/sd[O]-mn[O]));return m*sd[U]+mm[U],V*np.outer(sd[U],sd[U])
 A=np.eye(H);Btotal=np.zeros((H,H));bias=np.zeros(H);variance=np.zeros(H)
 for t in range(H):
  O=np.r_[np.arange(t),np.arange(H,H+t+1)];U,B,cc=conditional(mn,cn,O);F,V=propagate(cc);loc=int(np.flatnonzero(U==t)[0]);bb=F[loc]@B;fac=sd[t]*bb/sd[O];bias[t]=sd[t]*(F[loc]@mn[U]-bb@mn[O])-fac@mm[O]+mm[t];A[t,:t]=-fac[:t];Btotal[t,:t+1]=fac[t:];variance[t]=sd[t]**2*V[loc,loc]
 Ai=np.linalg.inv(A);return Ai@(bias+Btotal@u),(Ai*variance)@Ai.T
class OracleEPS(nn.Module):
 def __init__(self,mean,std,O):
  super().__init__();sd=np.repeat(std.reshape(-1),H);mm=np.repeat(mean.reshape(-1),H);mn=(mu-mm)/sd;cn=C/np.outer(sd,sd);U,B,cc=conditional(mn,cn,O);self.O=torch.tensor(O,device=device);self.U=torch.tensor(U,device=device);self.muO=torch.tensor(mn[O],device=device,dtype=torch.float32);self.muU=torch.tensor(mn[U],device=device,dtype=torch.float32);self.B=torch.tensor(B,device=device,dtype=torch.float32);self.inv=torch.tensor(np.stack([np.linalg.inv(a*cc+(1-a)*np.eye(len(U))) for a in sampler.alpha]),device=device,dtype=torch.float32);self.alpha=torch.tensor(sampler.alpha,device=device,dtype=torch.float32)
 def forward(self,x,info,t):
  j=int(t[0]);obs=x[:,0].reshape(len(x),-1);noisy=x[:,1].reshape(len(x),-1);cm=self.muU+(obs[:,self.O]-self.muO)@self.B.T;ep=(1-self.alpha[j]).sqrt()*(noisy[:,self.U]-self.alpha[j].sqrt()*cm)@self.inv[j].T;out=torch.zeros_like(noisy);out[:,self.U]=ep;return out.reshape(len(x),2,H)
def native(mean,std,u,operation):
 raw=np.zeros((N,2,H),dtype='float32');m=torch.zeros((N,2,H),device=device)
 with torch.inference_mode():
  if operation=='block':
   raw[:,1]=u;m[:,1]=1;O=np.arange(H,2*H);sampler.diffmodel=OracleEPS(mean,std,O);obs=torch.tensor((raw-mean)/std,device=device);return sampler.impute(obs,m,side(sampler,obs,m),1)[:,0,0].cpu().numpy()*std[0,0,0]+mean[0,0,0]
  x=np.empty((N,H),dtype='float32')
  for t in range(H):
   raw[:,1,:t+1]=u[:t+1];m[:,1,:t+1]=1
   if t:raw[:,0,:t]=x[:,:t];m[:,0,:t]=1
   O=np.r_[np.arange(t),np.arange(H,H+t+1)];sampler.diffmodel=OracleEPS(mean,std,O);obs=torch.tensor((raw-mean)/std,device=device);x[:,t]=sampler.impute(obs,m,side(sampler,obs,m),1)[:,0,0,t].cpu().numpy()*std[0,0,0]+mean[0,0,0]
  return x
def metric(m,V,tm,tC):
 R=sla.cholesky(tC,lower=True);S=sla.cholesky(V,lower=True);w=max(0.,float(np.sum((m-tm)**2)+np.trace(V)+np.trace(tC)-2*sla.svdvals(R.T@S).sum()));return dict(normalized_Gaussian_W2=float(np.sqrt(w/np.trace(tC))),variance_trace_ratio=float(np.trace(V)/np.trace(tC)),mean_MSE=float(np.mean((m-tm)**2)))
for corpus in f['corpora']:
 if corpus['mechanism']!='linear':continue
 assert sha(corpus['path'])==corpus['sha256'];z=np.load(corpus['path']);mean=z['train'].mean((0,2),keepdims=True);std=z['train'].std((0,2),keepdims=True)
 for pk in ['zero','sine']:
  u=np.zeros(H) if pk=='zero' else .4*np.sin(2*np.pi*(np.arange(H)+1)/8);L=np.eye(H)-.8*np.eye(H,k=-1);b=.7*u;b[0]+=.8*.25;Li=np.linalg.inv(L);md=Li@b;Cd=.25*Li@Li.T;U,B,Co=conditional(mu,C,np.arange(H,2*H));mo=mu[U]+B@(u-mu[H:])
  for operation in ['block','prefix']:
   m,V=law(mean,std,u,operation);key=f'{corpus["key"]}_{pk}_{operation}';art[key+'_mean']=m;art[key+'_cov']=V;rows.append(dict(key=key,training_normalization_seed=corpus['seed'],query=pk,operation=operation,do=metric(m,V,md,Cd),obs=metric(m,V,mo,Co),scope='Analytically propagated native50-step DDPM with exact Gaussian conditional noise predictor; privileged sampler attribution'))
   if corpus['seed']==2021:
    torch.manual_seed(65211);x=native(mean,std,u,operation);mean_z=float(np.max(np.abs(x.mean(0)-m)/np.sqrt(np.diag(V)/N)));cse=np.sqrt((np.outer(np.diag(V),np.diag(V))+V*V)/N);cov_z=float(np.max(np.abs(np.cov(x,rowvar=False,ddof=0)-V)/cse));assert mean_z<6 and cov_z<6,(key,mean_z,cov_z);checks.append(dict(key=key,native_samples=N,mean_max_standard_errors=mean_z,covariance_max_standard_errors=cov_z));art[key+'_native_samples']=x
p=paths()['artifact_root']/'NATIVE_ORACLE_SAMPLER65_V1.npz';np.savez_compressed(p,**art);dump(OUT/'NATIVE_ORACLE_SAMPLER65_V1.json',dict(utc=utc(),status='NATIVE_GAUSSIAN_SAMPLER_LAWS_QUALIFIED',rows=rows,checks=checks,artifact=str(p),artifact_sha256=sha(p),data_receipts=[{k:r[k] for k in ['key','path','sha256']} for r in f['corpora'] if r['mechanism']=='linear'],note_sha256=sha(ROOT/'NATIVE_SAMPLER_ORACLE_NOTE65_V1.md'),accepted_final_directions=0));print(json.dumps(dict(status='NATIVE_GAUSSIAN_SAMPLER_LAWS_QUALIFIED',rows=rows,checks=checks)))
