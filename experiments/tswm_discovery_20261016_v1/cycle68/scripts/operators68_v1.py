import sys,pathlib
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common68_v1 import *
def fit_ridge(X,Y,alpha):
 X=np.asarray(X,dtype='float64');Y=np.asarray(Y,dtype='float64');xm=X.mean(0);xs=X.std(0);xs[xs<1e-12]=1.;ym=Y.mean(0);Z=(X-xm)/xs;U,s,V=np.linalg.svd(Z,full_matrices=False);w=(V.T*(s/(s*s+alpha)))@(U.T@(Y-ym));return dict(xm=xm,xs=xs,ym=ym,w=w)
def apply(m,x):return ((x-m['xm'])/m['xs'])@m['w']+m['ym']
def ar_features(y,previous,u,degree):
 q=[y,previous,u]
 if degree==3:q.extend([y*y,y*y*y])
 return np.stack(q,axis=-1)
def raw_forecast(history,known,degree,alpha):
 mu,sd=norm(history);z=(history-mu)/sd;y=z[0];u=z[1];X=ar_features(y[1:-1],y[:-2],u[2:],degree);m=fit_ridge(X,y[2:,None],alpha);a=float(y[-1]);b=float(y[-2]);future=(known-mu[1,0])/sd[1,0];pred=[]
 for v in future:
  out=float(apply(m,ar_features(np.array(a),np.array(b),np.array(v),degree))[0]);pred.append(out);b,a=a,out
 pred=np.asarray(pred)*sd[0,0]+mu[0,0];return pred,m
def pca(E,k):
 E=np.asarray(E,dtype='float64');mu=E.mean(0);center=E-mu;vals,U=np.linalg.eigh(center@center.T);order=np.argsort(vals)[::-1][:k];s=np.sqrt(np.maximum(vals[order],0));assert s[-1]>1e-10;basis=(center.T@U[:,order]/s).T;return mu,basis,center@basis.T
def latent_forecast(E,history,known,k,alpha):
 mu,basis,C=pca(E,k);hm,hs=norm(history);u=(history[1]-hm[1,0])/hs[1,0];future=(known-hm[1,0])/hs[1,0];padded=np.r_[future,np.repeat(future[-1],(-len(future))%P)];control=u.reshape(-1,P);X=np.c_[C[:-1],control[1:]];m=fit_ridge(X,C[1:],alpha);current=C[-1];pred=[]
 for block in padded.reshape(-1,P):current=apply(m,np.r_[current,block]);pred.append(current@basis+mu)
 future_E=np.asarray(pred);return future_E,dict(mean=mu,basis=basis,operator=m,context_coefficients=C)
