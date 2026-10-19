"""Named public-data and pinned FlowNP dependency qualification; no model fitting."""
import sys,pathlib,json,hashlib,datetime,os,zipfile,io,subprocess,importlib.util
R=pathlib.Path(__file__).resolve().parents[1]; ROOT=R.parent
sys.path.insert(0,str(ROOT/'cycle29/scripts')); import runtime_profile
import requests
import numpy as np
import pandas as pd
O=R/'outputs'; A=R/'assets';O.mkdir(exist_ok=True);A.mkdir(exist_ok=True)
assert not (O/'intake.json').exists(), 'preserve existing attempt'
def sha(b):return hashlib.sha256(b).hexdigest()
def fetch(url,name):
    p=A/name
    if p.exists():raw=p.read_bytes(); return raw,{'url':url,'file':name,'sha256':sha(raw),'bytes':len(raw),'cache_reused':True}
    q=requests.get(url,timeout=(20,120));q.raise_for_status();raw=q.content
    p.write_bytes(raw);return raw,{'url':url,'file':name,'sha256':sha(raw),'bytes':len(raw),'http_status':q.status_code,'tls_verify':True}
found=[]
for folder in [ROOT/'upstream',ROOT/'cycle34/assets']:
    for path,dirs,files in os.walk(folder):
        dirs[:]=[d for d in dirs if d not in ['.git','__pycache__','node_modules']]
        for n in files:
            if 'etth1' in n.lower() or ('jena' in n.lower() and n.lower().endswith(('.csv','.zip'))):
                found.append(str(pathlib.Path(path,n).relative_to(ROOT)))
sources=[]
retrieval=json.loads((A/'retrieval.json').read_text());commit=retrieval['ett_commit']
raw,info=fetch(f'https://raw.githubusercontent.com/zhouhaoyi/ETDataset/{commit}/ETT-small/ETTh1.csv','ETTh1.csv');sources.append(info)
er=pd.read_csv(io.BytesIO(raw)); et=pd.DatetimeIndex(pd.to_datetime(er['date']));ey=er['OT'].to_numpy(dtype=np.float64)
frames=[]
for half in ['a','b']:
    z,info=fetch(f'https://www.bgc-jena.mpg.de/wetter/data/mpi_roof_2016{half}.zip',f'mpi_roof_2016{half}.zip');sources.append(info)
    with zipfile.ZipFile(io.BytesIO(z)) as f:
        names=[n for n in f.namelist() if n.lower().endswith('.csv')];assert len(names)==1,names
        df=pd.read_csv(f.open(names[0]),encoding='latin1',na_values=[-9999,-9999.0]);frames.append(df)
j=pd.concat(frames,ignore_index=True)
jt=pd.to_datetime(j['Date Time'],dayfirst=True);j.index=pd.DatetimeIndex(jt)
# A timestamp-exact subsample, not an interpolation or a centered mean.
j=j.loc[j.index.minute==0].sort_index();assert not j.index.duplicated().any()
jy=j['T (degC)'].to_numpy(dtype=np.float64);jt=j.index
records={}
for name,t,y,field,unit,license_text in [('ett',et,ey,'OT','physical unit not explicit in provider README; target is oil temperature','CC BY-ND 4.0'),('jena',jt,jy,'T (degC)','degrees Celsius','CC BY 4.0')]:
    cadence=np.diff(t.asi8)//1_000_000_000
    assert np.all(cadence==3600),(name,np.unique(cadence,return_counts=True))
    assert len(t)==len(y) and len(y)>8000
    a=int(.7*len(y));b=int(.85*len(y));finite=np.isfinite(y)
    mu=float(y[:a][finite[:a]].mean());sd=float(y[:a][finite[:a]].std());assert sd>0
    np.savez(A/(name+'.npz'),y=y,timestamps=t.asi8,normalized=(y-mu)/sd)
    splits={}
    for split,lo,hi in [('train',0,a),('val',a,b),('test',b,len(y))]:
        ends=np.arange(lo+192,hi-96+1)
        ends=np.array([e for e in ends if finite[e-192:e+96].all()],dtype=np.int64)
        # All history and targets must be inside that split. TEST indices only, no scores.
        np.save(A/(name+'_'+split+'_origins.npy'),ends)
        splits[split]={'start':lo,'stop_exclusive':hi,'complete_origins':len(ends),'history_and_future_inside_split':True}
    records[name]={'n':len(y),'field':field,'unit':unit,'license':license_text,'first':str(t[0]),'last':str(t[-1]),'missing':int((~finite).sum()),'cadence_seconds':3600,'splits':splits,'train_mean':mu,'train_std':sd,'array_sha256':sha((A/(name+'.npz')).read_bytes())}
vendor=R/'vendor';vendor.mkdir(exist_ok=True)
pip=[sys.executable,'-m','pip','install','--disable-pip-version-check','--target',str(vendor),'--no-deps','--no-index','--find-links',str(A),'attrdictionary==1.0.0','flow-matching==1.0.10']
q=subprocess.run(pip,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);(O/'dependency_install.log').write_text(q.stdout,encoding='utf-8');assert q.returncode==0,q.stdout[-2000:]
sys.path.insert(0,str(vendor));sys.path.insert(0,str(ROOT/'cycle25/upstream/flownp'))
from models.fnp import FNP
import torch
m=FNP(3,1,16,128,3,256,4,0.0,4,timesteps=8).cuda()
out=m.predict(torch.zeros(1,8,3,device='cuda'),torch.zeros(1,8,1,device='cuda'),torch.zeros(1,4,3,device='cuda'),num_samples=2).loc
assert out.shape==(2,1,4,1) and torch.isfinite(out).all()
report={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'retrieval':retrieval,'v1_failure':'server GitHub API connection refused before data download or training','prior_named_candidates':found,'sources':sources,'ett_commit':commit,'data':records,'flownp_commit':'f3403b6479dad33a7b93ba6edcf93c53d40f8e65','flownp_fnp_sha256':sha((ROOT/'cycle25/upstream/flownp/models/fnp.py').read_bytes()),'flownp_native_import_and_predict':True,'parameters':sum(p.numel() for p in m.parameters()),'test_scored':False}
(O/'intake.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2),flush=True)
