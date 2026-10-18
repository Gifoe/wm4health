from common67_v1 import *
class Baselines:
 def fit(self,train):
  self.xmean=train['features'].mean(0);self.xsd=train['features'].std(0);self.xsd[self.xsd==0]=1;self.Z=(train['features']-self.xmean)/self.xsd;self.A=np.c_[self.Z,np.ones(len(self.Z))]
  scales=np.array([scale_y(h) for h in train['history']]);self.Y=(train['target']-scales[:,0,None])/scales[:,1,None];self.mask=train['mask'];self.coef={};groups=collections.defaultdict(list)
  for j in range(H):groups[self.mask[:,j].tobytes()].append(j)
  for alpha in [1,100,10000]:
   coef=np.empty((self.A.shape[1],H))
   for key,cols in groups.items():
    good=np.frombuffer(key,dtype=np.bool_);A=self.A[good];pen=np.eye(A.shape[1])*alpha;pen[-1,-1]=0;coef[:,cols]=np.linalg.solve(A.T@A+pen,A.T@self.Y[np.ix_(good,cols)])
   assert np.isfinite(coef).all();self.coef[alpha]=coef
  p=artifact_root()/'RIDGE_AND_BANK67_V1.npz';assert not p.exists();np.savez_compressed(p,xmean=self.xmean,xsd=self.xsd,Z=self.Z,Y=self.Y,mask=self.mask,**{'ridge_'+str(k):v for k,v in self.coef.items()});dump(OUT/'BASELINE_FIT67_V1.json',dict(utc=utc(),train_cases=len(self.Z),ridge_configurations=3,alphas=[1,100,10000],knn_configurations=[1,4,16],train_labels_only=True,artifact=str(p),sha256=sha(p),ridge_coefficients_per_configuration=int(coef.size)))
 def predict(self,history,known):
  mu,sd=scale_y(history);z=(feature(history,known)-self.xmean)/self.xsd;order=np.argsort(((self.Z-z)**2).mean(1),kind='stable');out={'context_mean':np.full(H,mu),'last_value':np.full(H,history[0,-1]),'seasonal_day':history[0,-1440:-720].copy()}
  for k in [1,4,16]:
   ix=order[:k];n=self.mask[ix].sum(0);y=np.divide(np.where(self.mask[ix],self.Y[ix],0).sum(0),n,out=np.zeros(H),where=n>0);out['analog_k'+str(k)]=y*sd+mu
  for alpha,coef in self.coef.items():out['ridge_alpha'+str(alpha)]=np.r_[z,1]@coef*sd+mu
  return out
