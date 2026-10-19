"""Batched numeric implementation of existing linear-Gaussian controlled EM; not a new method."""
import runtime_profile
import numpy as np

def evolution(p,u,implicit=False):
 full=np.eye(5)+(-1 if implicit else 1)*np.einsum('nti,ijk->ntjk',u,p['V_matrices'])
 if implicit:full=np.linalg.inv(full)
 return full

def estep(p,y,u,implicit=False):
 n,times=y.shape[:2];full=evolution(p,u,implicit);a=full[:,:,1:,1:];b=full[:,:,1:,0]
 c=p['C'][0];h=p['h'][0];r=p['Sig_w'][0,0];q=p['Sig_v']
 fm=np.zeros((n,times,4));fp=np.zeros((n,times,4,4));predm=np.zeros_like(fm);predp=np.zeros_like(fp)
 mean=np.repeat(p['mu_0'][None],n,axis=0);cov=np.repeat(p['Sig_0'][None],n,axis=0);likelihood=0.
 for t in range(times):
  predm[:,t]=mean;predp[:,t]=cov
  residual=y[:,t,0]-h-mean@c;pc=cov@c;s=pc@c+r
  assert np.min(s)>0
  likelihood-=.5*float(np.sum(np.log(2*np.pi*s)+residual**2/s))
  gain=pc/s[:,None];mean=mean+gain*residual[:,None];cov=cov-gain[:,:,None]*pc[:,None,:]
  if t>0:cov=.5*(cov+cov.transpose(0,2,1))
  fm[:,t]=mean;fp[:,t]=cov
  if t<times-1:
   mean=(a[:,t]@mean[:,:,None])[:,:,0]+b[:,t]
   cov=q+a[:,t]@cov@a[:,t].transpose(0,2,1)
   cov=.5*(cov+cov.transpose(0,2,1))
 sm=fm.copy();sp=fp.copy();cross=np.zeros((n,times-1,4,4))
 for t in range(times-2,-1,-1):
  gain=fp[:,t]@np.linalg.solve(predp[:,t+1],a[:,t]).transpose(0,2,1)
  sm[:,t]=fm[:,t]+(gain@(sm[:,t+1]-predm[:,t+1])[:,:,None])[:,:,0]
  cross[:,t]=gain@sp[:,t+1]
  cov=fp[:,t]+gain@(sp[:,t+1]-predp[:,t+1])@gain.transpose(0,2,1)
  sp[:,t]=.5*(cov+cov.transpose(0,2,1))
 return sm,sp,cross,likelihood

def mstep(p,y,u,m,cov,cross,corrected=False):
 n,times=m.shape[:2];mu=m[:,:-1];dm=m[:,1:]-mu;length=n*(times-1)
 g=np.zeros((n,times-1,5,5));g[:,:,0,0]=1;g[:,:,0,1:]=mu;g[:,:,1:,0]=mu
 g[:,:,1:,1:]=cov[:,:-1]+mu[:,:,:,None]*mu[:,:,None,:]
 h=np.zeros((n,times-1,4,5));h[:,:,:,0]=dm
 h[:,:,:,1:]=(cross.transpose(0,1,3,2) if corrected else cross)-cov[:,:-1]+dm[:,:,:,None]*mu[:,:,None,:]
 uf=u.reshape(length,2);gf=g.reshape(length,5,5);hf=h.reshape(length,4,5)
 gram=np.einsum('ni,nj,nab->iajb',uf,uf,gf,optimize=True).reshape(10,10)
 target=np.einsum('ni,nab->aib',uf,hf,optimize=True).reshape(4,10)
 coefficients=np.linalg.solve(gram,target.T).T
 new={k:v.copy() for k,v in p.items()};v=np.zeros((2,5,5))
 for i in range(2):v[i,1:]=coefficients[:,5*i:5*(i+1)]
 new['V_matrices']=v;full=evolution(new,u);a=full[:,:,1:,1:];b=full[:,:,1:,0]
 error=m[:,1:]-(a@mu[:,:,:,None])[:,:,:,0]-b;ac=a@cross
 new['Sig_v']=(cov[:,1:]-ac-ac.transpose(0,1,3,2)+a@cov[:,:-1]@a.transpose(0,1,3,2)+error[:,:,:,None]*error[:,:,None,:]).mean(axis=(0,1))
 c=p['C'][0];new['h']=np.array([(y[:,:,0]-m@c).mean()]);residual=y[:,:,0]-m@c-new['h'][0]
 new['Sig_w']=np.array([[np.mean((cov@c)@c+residual**2)]])
 new['mu_0']=m[:,0].mean(0);center=m[:,0]-new['mu_0']
 new['Sig_0']=(cov[:,0]+center[:,:,None]*center[:,None,:]).mean(0)
 return new

def initial(seed,u,reflect):
 rng=np.random.RandomState(seed);a0=reflect(rng.randn(4,4)/2);a1=reflect(rng.randn(5,5)/np.sqrt(5));v=np.zeros((2,5,5))
 v[0,1:,1:]=.5*(a0-np.eye(4))/.01;v[1,1:]=.5*a1[1:]/(.01*np.max(np.abs(u[:,:,1])))
 return dict(V_matrices=v,C=np.array([[1.,0,0,0]]),h=np.zeros(1),Sig_v=.1*np.eye(4),Sig_w=.1*np.eye(1),mu_0=np.zeros(4),Sig_0=np.eye(4))
