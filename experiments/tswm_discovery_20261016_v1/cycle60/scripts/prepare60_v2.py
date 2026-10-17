"""Preserve complete-row/chronological splits and fit-only scales; never copy old artifacts."""
from common60 import *
import numpy as np
from scipy.io import loadmat
def main():
    source=source_check();p=paths();old=PROJECT/'cycle59/outputs';oldsource=read(old/'DOWNLOAD59_V2.json')
    raw={}
    for name in ['B','H']:
        row=next(r for r in oldsource['files'] if r['path'].endswith(f'N87_5_{name}.csv.gz'));assert sha(row['server_path'])==row['sha256']
        raw[name]=np.loadtxt(row['server_path'],delimiter=',')
    split=read(old/'SPLIT59_V1.json')['rows'];cases={}
    cases['N87']={k:(raw['B'][v,:,None],raw['H'][v,:,None]) for k,v in split.items()}
    matrow=next(r for r in source['files'] if r['kind']=='data');mat=loadmat(matrow['server_path'])
    t=np.asarray(mat['t']).reshape(-1);u=np.asarray(mat['u']).reshape(-1,1);y=np.asarray(mat['y']).reshape(-1,1)
    assert len(t)==len(u)==len(y) and len(t)>50000 and np.isfinite(u).all() and np.isfinite(y).all()
    dt=float(np.median(np.diff(t)));assert dt>0 and np.max(abs(np.diff(t)-dt))<max(1e-9*dt,1e-10)
    cases['pick_place']={k:(u[a:b:10][None],y[a:b:10][None]) for k,a,b in [('train',0,40000),('validation',40000,50000),('test',50000,len(t))]}
    manifest={'utc':utc(),'status':'TWO_SYSTEM_CAUSAL_INPUT_AND_FIT_ONLY_NORMALIZATION_PREPARED','source_commit':source['source_commit'],'source_receipt_sha256':sha(OUT/'DOWNLOAD60_V1.json'),'protocol_sha256':sha(ROOT/'PROTOCOL60_V1.md'),'script_sha256':sha(__file__),'systems':{}}
    for system,parts in cases.items():
        folder=p['data_root']/'prepared_v2'/system;folder.mkdir(exist_ok=False,parents=True)
        ut,yt=parts['train'];mu=float(ut.mean());su=float(ut.std());my=float(yt.mean());sy=float(yt.std());assert su>0 and sy>0
        entries=[]
        for name,(uu,yy) in parts.items():
            un=(uu-mu)/su;du=np.diff(un,axis=1,prepend=un[:,:1]);inputs=np.concatenate([un,du],axis=-1).astype('float32');targets=((yy-my)/sy).astype('float32')
            assert inputs.shape[1]>228 and targets.shape[:-1]==inputs.shape[:-1]
            dest=folder/(name+'.npz')
            with dest.open('xb') as f:np.savez(f,u=inputs,y=targets)
            entries.append({'split':name,'u_shape':list(inputs.shape),'y_shape':list(targets.shape),'file':str(dest),'sha256':sha(dest)})
        manifest['systems'][system]={'normalization':{'u_mean':mu,'u_std':su,'y_mean':my,'y_std':sy,'fit_only':True},'splits':entries,'physical_dt_seconds':62.5e-9 if system=='N87' else dt*10,'test_prefix':100,'split_semantics':'C59 complete-row grouping; historical test scores known' if system=='N87' else 'Raw chronological 0:40000 /40000:50000 /50000:end, source decimation every tenth point; custom prefix100 forecast protocol','source_raw_sha256':matrow['sha256'] if system=='pick_place' else [r['sha256'] for r in oldsource['files'] if r['kind']=='data'],'raw_shape':[len(t),1] if system=='pick_place' else list(raw['B'].shape)}
    dump(OUT/'DATA60_V2.json',manifest)
    print(json.dumps({'status':manifest['status'],'systems':{k:{'dt':v['physical_dt_seconds'],'shapes':[r['u_shape'] for r in v['splits']]} for k,v in manifest['systems'].items()}}),flush=True)
if __name__=='__main__':main()
