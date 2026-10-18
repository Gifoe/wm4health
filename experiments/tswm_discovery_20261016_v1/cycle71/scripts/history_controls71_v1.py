"""Ordinary history-fitted affine residual autoregression; no future input."""
import numpy as np

ALPHAS=(1e-8,1e-6,1e-4,1e-2,1.,100.)

def fit(history,mu,scale,lag,alpha):
 x=np.asarray(history,dtype=np.float64).reshape(len(history),-1)
 mu=np.asarray(mu,dtype=np.float64);scale=np.asarray(scale,dtype=np.float64)
 assert lag in (1,2) and len(x)>lag and alpha>0 and np.all(scale>0)
 assert x.shape[1:]==mu.shape==scale.shape and np.isfinite(x).all()
 z=(x-mu)/scale
 features=np.concatenate([z[lag-1-j:len(z)-1-j] for j in range(lag)]+[np.ones((len(z)-lag,1))],axis=1)
 delta=z[lag:]-z[lag-1:-1]
 penalty=np.eye(features.shape[1]);penalty[-1,-1]=0
 n=len(features)
 coef=np.linalg.solve(features.T@features/n+alpha*penalty,features.T@delta/n)
 assert np.isfinite(coef).all()
 return dict(coef=coef,lag=lag,mu=mu.copy(),scale=scale.copy(),last=z[-lag:].copy(),condition_number=float(np.linalg.cond(features.T@features/n+alpha*penalty)))

def predict(model,horizon):
 states=[z.copy() for z in model['last']];out=[]
 with np.errstate(over='raise',invalid='raise'):
  for _ in range(horizon):
   features=np.concatenate([states[-1-j] for j in range(model['lag'])]+[np.ones(1)])
   nxt=states[-1]+features@model['coef'];assert np.isfinite(nxt).all()
   states.append(nxt);out.append(nxt*model['scale']+model['mu'])
 return np.asarray(out,dtype=np.float64)

def qualify():
 # Known affine law; independent augmented least squares uses a different solve.
 rng=np.random.default_rng(2026730);d=4
 A=np.array([[.8,.1,0,0],[0,.7,.1,0],[0,0,.6,.1],[.1,0,0,.5]])
 b=np.array([.1,-.03,.04,-.06]);x=[rng.normal(size=d)]
 for i in range(79):x.append(A@x[-1]+b+rng.normal(scale=.02,size=d))
 x=np.asarray(x);mu=np.array([1.,-2.,.4,.8]);scale=np.array([.3,2.,.8,1.2]);checks=[]
 for lag in (1,2):
  m=fit(x[:49],mu,scale,lag,.01);z=(x[:49]-mu)/scale;rows=[];targets=[]
  for t in range(lag-1,48):rows.append(np.concatenate([z[t-j] for j in range(lag)]+[np.ones(1)]));targets.append(z[t+1]-z[t])
  X=np.asarray(rows);Y=np.asarray(targets);root=np.sqrt(len(X)*.01)*np.eye(X.shape[1]);root[-1,-1]=0
  independent=np.linalg.lstsq(np.vstack([X,root]),np.vstack([Y,np.zeros((X.shape[1],d))]),rcond=None)[0]
  coefficient_error=float(np.max(np.abs(independent-m['coef'])))
  ref=list(z[-lag:].copy());pred=[]
  for _ in range(25):
   value=ref[-1]+np.concatenate([ref[-1-j] for j in range(lag)]+[np.ones(1)])@independent;ref.append(value);pred.append(value*scale+mu)
  error=float(np.max(np.abs(np.asarray(pred)-predict(m,25))))
  assert coefficient_error<1e-10 and error<1e-10,(lag,coefficient_error,error)
  checks.append(dict(lag=lag,coefficient_max_abs=coefficient_error,rollout_max_abs=error))
 return dict(status='PASS_INDEPENDENT_AFFINE_REGRESSION_AND_RECURSION',checks=checks,performance_evidence=False)
