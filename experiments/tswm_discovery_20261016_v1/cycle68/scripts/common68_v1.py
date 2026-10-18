import os
os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='4'
import pathlib,json,hashlib,datetime
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];PROJECT=ROOT.parent;OUT=ROOT/'outputs';OUT.mkdir(exist_ok=True,parents=True);L=3200;H=100;P=16
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  while b:=f.read(8*1024*1024):h.update(b)
 return h.hexdigest()
def dump(p,obj):
 p=pathlib.Path(p);p.parent.mkdir(exist_ok=True,parents=True)
 with p.open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,allow_nan=False)
def state(name,obj):
 p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(dict(utc=utc(),**obj),indent=2,allow_nan=False),encoding='utf-8');t.replace(p)
def storage(kind):
 p=pathlib.Path(read(PROJECT/'STORAGE_LOCATION_V1.json')[kind+'_root'])/'cycle68';assert p.drive.upper()=='G:';p.mkdir(exist_ok=True,parents=True);return p
def generate(seed,family):
 rng=np.random.default_rng(seed);r=rng.uniform(.94,.98);theta=rng.uniform(.08,.18);gain=rng.uniform(.7,1.3);c=0. if family=='linear' else rng.uniform(.0002,.0010);a1=2*r*np.cos(theta);a2=-r*r;b=gain*(1-a1-a2)
 burn=1024;n=burn+L+H+2;u=np.repeat(rng.uniform(-1,1,(n+15)//16),16)[:n];y=np.zeros(n)
 for t in range(1,n-1):y[t+1]=a1*y[t]+a2*y[t-1]+b*u[t]-c*y[t]**3
 assert np.isfinite(y).all() and abs(y).max()<100
 start=burn;raw=np.stack([y[start:start+L+H],u[start-1:start+L+H-1]])
 return raw,dict(seed=seed,family=family,r=float(r),theta=float(theta),gain=float(gain),c=float(c),a1=float(a1),a2=float(a2),b=float(b),burn_in=burn,alignment='raw[1,t]=u_original[burn+t-1]; raw[0,t]=y_original[burn+t]')
def norm(history):
 mu=history.mean(-1,keepdims=True);sd=history.std(-1,keepdims=True,ddof=1);sd[sd==0]=1.;return mu,sd+1e-8
def metric(pred,target,history):
 pred=np.asarray(pred,dtype='float64');assert pred.shape==(H,) and np.isfinite(pred).all();d=pred-np.asarray(target,dtype='float64');sd=float(norm(history)[1][0,0]);return dict(MAE=float(abs(d).mean()),RMSE=float(np.sqrt((d*d).mean())),normalized_MAE=float(abs(d).mean()/sd),first50_normalized_MAE=float(abs(d[:50]).mean()/sd),last50_normalized_MAE=float(abs(d[50:]).mean()/sd))
