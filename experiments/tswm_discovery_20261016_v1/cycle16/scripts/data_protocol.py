"""Causal calendar-only forecasting adapter; held-out test labels are not exported."""
import hashlib,json,pathlib,math
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
L=151;H=75;T=L+H-1;LAGS=np.asarray(list(range(1,25))+[48,72,96,120,144])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def prepare():
 import pandas as pd
 out=ROOT/'outputs/data';out.mkdir(parents=True,exist_ok=True);assert not (out/'registry.json').exists()
 path=pathlib.Path('D:/baseline_reproduction_seed2021_v1/data/NP.csv')
 old=json.loads((ROOT.parent/'outputs/existing_future_support.json').read_text())['provenance']
 assert sha(path)==next(r for r in old if r['dataset']=='NP')['data_sha256']
 frame=pd.read_csv(path);frame['date']=pd.to_datetime(frame['date']);wide=frame.pivot(index='date',columns='cols',values='data')
 N=len(wide);train_end=int(int(N*.8)*.875);val_end=int(N*.8)
 times=wide.index[:val_end];assert times.is_monotonic_increasing and np.all(np.diff(times.asi8)==3600*10**9)
 raw=wide['OT'].iloc[:val_end].to_numpy(dtype=np.float64);assert np.isfinite(raw).all()
 mean=float(raw[:train_end].mean());std=float(raw[:train_end].std());assert std>1e-6
 y=((raw-mean)/std).astype(np.float32)
 hour=times.hour.to_numpy();weekday=times.dayofweek.to_numpy();day=times.dayofyear.to_numpy()-1
 u=np.stack([np.sin(2*np.pi*hour/24),np.cos(2*np.pi*hour/24),np.sin(2*np.pi*weekday/7),np.cos(2*np.pi*weekday/7),
             np.sin(2*np.pi*day/365.2425),np.cos(2*np.pi*day/365.2425)],axis=1).astype(np.float32)
 train=np.arange(L-1,train_end-H,dtype=np.int64);val=np.arange(train_end-1,val_end-H,H,dtype=np.int64)
 assert np.all(train+H<train_end) and np.all(val+1>=train_end) and np.all(val+H<val_end)
 assert np.all(np.diff(val)>=H)
 arrays=out/'NP_trainval.npz';np.savez_compressed(arrays,y=y,u=u,train_origins=train,val_origins=val,mean=mean,std=std)
 receipt={'dataset':'NP','source_csv':str(path),'source_sha256':sha(path),'N_original':N,'train_end':train_end,'val_end':val_end,
          'test_labels_exported':False,'test_labels_evaluated':False,'prepared_path':arrays.relative_to(ROOT).as_posix(),'prepared_sha256':sha(arrays),
          'history':L,'horizon':H,'source_steps':T,'scored_source_outputs':[L-1,T],'train_origins':len(train),'validation_origins':len(val),
          'validation_targets_nonoverlap':True,'target':'OT','target_mean_train_only':mean,'target_std_train_only':std,
          'conditions':['sin_hour','cos_hour','sin_weekday','cos_weekday','sin_day_of_year','cos_day_of_year'],
          'conditions_available_at_origin':'Deterministic future timestamps only','raw_covariates_used':False,
          'raw_covariates_excluded':[str(c) for c in wide.columns if c!='OT'],'normalization_scope':'training OT rows only',
          'first_training_timestamp':str(times[0]),'last_validation_timestamp':str(times[-1]),
          'protocol_sha256':sha(ROOT/'FULL_MODEL_PLAN.md'),'adapter_sha256':sha(pathlib.Path(__file__))}
 (out/'registry.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps(receipt),flush=True)
class Data:
 def __init__(self):
  self.registry=json.loads((ROOT/'outputs/data/registry.json').read_text());path=ROOT/self.registry['prepared_path']
  assert sha(path)==self.registry['prepared_sha256']
  with np.load(path,allow_pickle=False) as a:
   self.y=a['y'];self.u=a['u'];self.train=a['train_origins'];self.val=a['val_origins'];self.std=float(a['std']);self.mean=float(a['mean'])
  assert len(self.y)==self.registry['val_end']
 def batch(self,origins,partition):
  origins=np.asarray(origins,dtype=np.int64);assert partition in ['train','val']
  assert np.all(origins>=L-1) and np.all(origins+H<len(self.y))
  if partition=='train':assert np.all(origins+H<self.registry['train_end'])
  else:assert np.all(origins+1>=self.registry['train_end'])
  idx=origins[:,None]-(L-1)+np.arange(T)[None,:]
  obs=self.y[idx,None].copy();obs[:,L:]=0
  # Source next-state prediction at output indexL-1 targets origin+1.
  cond=self.u[idx+1].copy();target=self.y[origins[:,None]+np.arange(1,H+1)[None,:],None].copy()
  mask=np.zeros((len(origins),T,1),dtype=bool);mask[:,:L]=True
  assert np.all(idx[:,-1]+1==origins+H) and np.all(obs[:,L:]==0)
  np.testing.assert_array_equal(obs[:,L-1,0],self.y[origins])
  np.testing.assert_array_equal(target[:,0,0],self.y[origins+1])
  return obs,cond,mask,target
def metrics(mean,var,target):
 mean=np.asarray(mean,dtype=np.float64);var=np.asarray(var,dtype=np.float64);target=np.asarray(target,dtype=np.float64)
 assert mean.shape==var.shape==target.shape and np.isfinite(mean).all() and np.isfinite(var).all() and (var>0).all()
 error=target-mean;nll=.5*(np.log(2*np.pi*var)+error**2/var)
 return {'NLL':float(nll.mean()),'MAE_standardized':float(np.abs(error).mean()),'RMSE_standardized':float(np.sqrt((error**2).mean())),
         'coverage95':float((np.abs(error)<=1.959963984540054*np.sqrt(var)).mean()),'mean_predicted_variance':float(var.mean())}
def references():
 data=Data();out=ROOT/'outputs/references';out.mkdir(parents=True,exist_ok=True);assert not (out/'results.json').exists()
 y,u=data.y.astype(np.float64),data.u.astype(np.float64);end=data.registry['train_end'];roots=np.arange(L-1,end-H,H)
 def simple(origins,kind):
  if kind=='persistence':return np.repeat(y[origins,None],H,axis=1)
  return y[origins[:,None]-23+(np.arange(H)%24)[None,:]]
 truth=y[roots[:,None]+np.arange(1,H+1)[None,:]];valtruth=y[data.val[:,None]+np.arange(1,H+1)[None,:]]
 arrays={'validation_origins':data.val,'target':valtruth};rows=[]
 for kind in ['persistence','seasonal24']:
  variance=np.maximum(np.mean((truth-simple(roots,kind))**2,axis=0),1e-6)
  prediction=simple(data.val,kind);vv=np.broadcast_to(variance,prediction.shape)
  arrays[kind+'_mean']=prediction;arrays[kind+'_variance']=vv
  rows.append({'model':kind,**metrics(prediction,vv,valtruth),'variance_source':'Training-only residual MSE per horizon; no validation recalibration'})
 t=np.arange(int(LAGS.max()),end);X=np.concatenate([y[t[:,None]-LAGS[None,:]],u[t],np.ones((len(t),1))],axis=1)
 penalty=np.eye(X.shape[1]);penalty[-1,-1]=0
 beta=np.linalg.solve(X.T@X+penalty,X.T@y[t]);residual=y[t]-X@beta;sigma2=float(np.dot(residual,residual)/(len(t)-len(beta)))
 psi=np.zeros(H);psi[0]=1
 for j in range(1,H):psi[j]=sum(beta[k]*psi[j-int(lag)] for k,lag in enumerate(LAGS) if j>=lag)
 variance=sigma2*np.cumsum(psi**2);prediction=[]
 for origin in data.val:
  history=list(y[origin-L+1:origin+1]);one=[]
  for h in range(1,H+1):
   features=np.r_[[history[-int(lag)] for lag in LAGS],u[origin+h],1.]
   value=float(features@beta);one.append(value);history.append(value)
  prediction.append(one)
 prediction=np.asarray(prediction);vv=np.broadcast_to(variance,prediction.shape)
 rows.append({'model':'Ridge_ARX','ridge_alpha':1.,'coefficients':len(beta),**metrics(prediction,vv,valtruth),
              'variance_source':'Train residual Gaussian innovation; exact linear AR impulse variance; ignores coefficient uncertainty'})
 arrays.update(arx_mean=prediction,arx_variance=vv,arx_beta=beta,arx_sigma2=np.asarray(sigma2),arx_lags=LAGS)
 np.savez_compressed(out/'arrays.npz',**arrays)
 (out/'results.json').write_text(json.dumps({'rows':rows,'validation_only':True,'no_test_evaluation':True},indent=2),encoding='utf-8');print(json.dumps(rows),flush=True)
