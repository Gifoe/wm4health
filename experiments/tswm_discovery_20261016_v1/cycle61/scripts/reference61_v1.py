"""Independent NumPy expanded-state probabilities; no native inference calls."""
import numpy as np

def matrix(duration,transition):
 K,D=duration.shape;S=np.flip(np.cumsum(np.flip(duration,axis=-1),axis=-1),axis=-1);P=np.zeros((K*D,K*D))
 for z in range(K):
  for a in range(D):
   hazard=duration[z,a]/S[z,a]
   if a+1<D:P[z*D+a,z*D+a+1]=S[z,a+1]/S[z,a]
   for w in range(K):P[z*D+a,w*D]+=hazard*transition[z,a,w]
 assert np.allclose(P.sum(1),1,atol=1e-10)
 return P,S

def emission(x,mean,variance):
 return -.5*np.sum((x[:,:,None,:]-mean[None,None])**2/variance[None,None]+np.log(variance)[None,None]+np.log(2*np.pi),axis=-1)

def filter_all(x,initial,duration,transition,mean,variance):
 P,S=matrix(duration,transition);B,T,_=x.shape;K,D=duration.shape;q=np.zeros((B,K*D));q[:,::D]=initial;post=[];ll=[];total=np.zeros(B);lp=emission(x,mean,variance)
 for t in range(T):
  if t:q=q@P
  peak=lp[:,t].max(-1);q=q*np.repeat(np.exp(lp[:,t]-peak[:,None]),D,axis=-1);mass=q.sum(-1);assert np.all(mass>0);total=total+np.log(mass)+peak;q=q/mass[:,None];post.append(q.reshape(B,K,D).copy());ll.append(total.copy())
 return np.stack(post,1),np.stack(ll,1),P,S

def survival(q,S,horizons):
 B,K,D=q.shape;out=np.zeros((B,len(horizons)))
 for j,h in enumerate(horizons):
  for z in range(K):
   for a in range(max(0,D-h)):out[:,j]+=q[:,z,a]*S[z,a+h]/S[z,a]
 return out

def marginal_means(q,P,mean,horizon):
 B,K,D=q.shape;q=q.reshape(B,-1);out=[]
 for h in range(horizon):q=q@P;out.append(q.reshape(B,K,D).sum(-1)@mean)
 return np.stack(out,1)
