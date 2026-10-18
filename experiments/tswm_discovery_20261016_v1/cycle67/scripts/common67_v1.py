import os
os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='4'
import pathlib,json,hashlib,datetime,csv,collections
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];PROJECT=ROOT.parent;OUT=ROOT/'outputs';OUT.mkdir(exist_ok=True,parents=True);L=2880;H=720
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def dump(p,obj):
 p=pathlib.Path(p);p.parent.mkdir(exist_ok=True,parents=True)
 with p.open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,allow_nan=False)
def state(name,obj):
 p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(dict(utc=utc(),**obj),indent=2,allow_nan=False),encoding='utf-8');t.replace(p)
def artifact_root():
 p=pathlib.Path(read(PROJECT/'STORAGE_LOCATION_V1.json')['artifact_root'])/'cycle67';p.mkdir(exist_ok=True,parents=True);return p
def numeric(v):
 try:return float(v)
 except (ValueError,TypeError):return float('nan')
def csv_record(rec):
 assert sha(rec['path'])==rec['sha256']
 with pathlib.Path(rec['path']).open(encoding='utf-8-sig',newline='') as f:
  reader=csv.DictReader(f);reader.fieldnames=[x.strip() for x in reader.fieldnames];rows=list(reader)
 t=np.array([np.datetime64(datetime.datetime.fromisoformat(r['Timestamp'])) for r in rows],dtype='datetime64[ns]');names=['Libre GL','Dexcom GL','HR','Carbs','Protein','Fat'];values={n:np.array([numeric(r.get(n)) for r in rows],dtype='float64') for n in names};meal=np.array([bool(r.get('Meal Type','').strip()) for r in rows]);return t,values,meal
def ffill15(x):
 x=x.copy();last=float('nan');gap=0
 for i,v in enumerate(x):
  if np.isfinite(v):last=v;gap=0
  else:
   gap+=1
   if gap<=15:x[i]=last
 return x
def scale_y(history):
 y=np.asarray(history[0],dtype='float64');sd=y.std(ddof=1);return y.mean(),(1. if sd==0 else sd)+1e-8
def feature(history,known):
 mu,sd=scale_y(history);return np.r_[((history[0]-mu)/sd).reshape(96,30).mean(1),known.reshape(3,12,60).sum(2).flatten()].astype('float64')
def metric(pred,target,mask,history):
 pred=np.asarray(pred,dtype='float64');truth=np.asarray(target,dtype='float64');assert pred.shape==truth.shape==(H,) and np.isfinite(pred).all();d=pred[mask]-truth[mask];_,sd=scale_y(history);return dict(MAE_mg_dL=float(np.abs(d).mean()),RMSE_mg_dL=float(np.sqrt((d*d).mean())),context_normalized_MAE=float(np.abs(d).mean()/sd),scored_points=int(mask.sum()))
