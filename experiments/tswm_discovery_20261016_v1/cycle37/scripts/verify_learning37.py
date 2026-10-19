"""Independent raw/window/score/replay audit; never imports production score/data code."""
import pathlib,sys,json,hashlib,re,time,datetime
import numpy as np
import pandas as pd
R=pathlib.Path(__file__).resolve().parents[1];B=R.parent
sys.path.insert(0,str(R/'scripts'))
# Only native architecture construction/runtime are shared, not windows or scoring.
from common37_v2 import make_model,seed,state_sha
import torch,joblib
O=R/'outputs/learning_v2';dest=R/'outputs/verification_v1.json'
assert not dest.exists();start=time.monotonic()
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
result=json.loads((O/'results.json').read_text(encoding='utf-8'));assert result['status']=='completed' and not result['test_scored']
assert result['protocol_sha256']==sha(R/'LEARNING_PROTOCOL.md') and result['common_sha256']==sha(R/'scripts/common37_v2.py')
assert result['runner_sha256']==sha(R/'scripts/run_learning37_v2.py')
assert result['raw_archive_sha256']==sha(R/'data_intake/GEFCom2014_original.zip')
def independent_data(name,meta):
    if name=='wind':
        raw=pd.read_csv(R/'data/wind_zone1_original.csv');target='TARGETVAR';w=['U10','V10','U100','V100']
    else:
        y=pd.read_csv(R/'data/solar_train15_original.csv');x=pd.read_csv(R/'data/solar_predictors15_original.csv')
        w=[k for k in x if re.fullmatch(r'VAR\d+',k)];assert len(w)==12
        raw=y[y.ZONEID==1].merge(x.loc[x.ZONEID==1,['ZONEID','TIMESTAMP']+w],on=['ZONEID','TIMESTAMP'],validate='one_to_one');target='POWER'
    raw['time']=pd.to_datetime(raw.TIMESTAMP.str.strip(),format='%Y%m%d %H:%M');raw=raw.sort_values('time').set_index('time')
    can=pd.read_csv(R/'data_v3'/f'{name}_zone1_canonical.csv',parse_dates=['valid_time']).set_index('valid_time')
    assert np.array_equal(raw.index.values,can.index.values) and w==meta['weather_columns']
    assert np.allclose(raw[target],can.y,atol=1e-12,rtol=1e-12,equal_nan=True)
    assert np.allclose(raw[w],can[w],atol=1e-12,rtol=1e-12,equal_nan=True)
    assert sha(R/'data_v3'/f'{name}_zone1_canonical.csv')==meta['canonical_sha256']
    origins=pd.DatetimeIndex(sorted((can.index-pd.Timedelta('1h')).floor('D').unique()));n=len(origins)
    a,b,c=[int(n*x) for x in [.56,.70,.85]];ranges={'fit':(0,a-1),'cal':(a,b-1),'val':(b,c-1),'test':(c,n)}
    def cal(t):
        t=pd.DatetimeIndex(t);hr=t.hour.to_numpy()/24;wk=(t.dayofweek.to_numpy()+hr)/7;yr=(t.dayofyear.to_numpy()-1+hr)/365.25
        return np.column_stack([np.sin(2*np.pi*hr),np.cos(2*np.pi*hr),np.sin(2*np.pi*wk),np.cos(2*np.pi*wk),np.sin(2*np.pi*yr),np.cos(2*np.pi*yr)])
    groups={};counts={}
    for part,(lo,hi) in ranges.items():
        rec=[]
        for o in origins[lo:hi]:
            h=pd.date_range(o-pd.Timedelta(hours=168),o-pd.Timedelta(hours=1),freq='h')
            t=pd.date_range(o+pd.Timedelta(hours=1),o+pd.Timedelta(hours=24),freq='h')
            hd=can.reindex(h);td=can.reindex(t)
            hy=hd.y.to_numpy()[:,None];hx=np.column_stack([hd[w].values,cal(h)]);fx=np.column_stack([td[w].values,cal(t)]);y=td.y.to_numpy()
            if not all(np.isfinite(v).all() for v in [hy,hx,fx,y]):continue
            rec.append((str(o),hy,hx,fx,y))
        g={'origins':np.array([x[0] for x in rec]),'hy':np.stack([x[1] for x in rec]),'hx':np.stack([x[2] for x in rec]),
           'fx':np.stack([x[3] for x in rec]),'y':np.stack([x[4] for x in rec])};groups[part]=g
        assert len(rec)==meta['counts'][part]['retained'];counts[part]=len(rec)
        with np.load(O/name/'registered_origins.npz') as z:assert np.array_equal(z[part],g['origins'])
    fit=groups['fit'];xx=np.concatenate([fit['hx'].reshape(-1,len(w)+6),fit['fx'].reshape(-1,len(w)+6)])
    xm=xx.mean(0);xs=np.maximum(xx.std(0),1e-6);ym=fit['y'].mean();ys=max(fit['y'].std(),1e-6)
    for key,v in [('x_mean',xm),('x_std',xs),('y_mean',ym),('y_std',ys)]:assert np.allclose(v,meta['scaler'][key],atol=1e-10,rtol=1e-12)
    for g in groups.values():
        g['hy']=((g['hy']-ym)/ys).astype('float32');g['hx']=((g['hx']-xm)/xs).astype('float32');g['fx']=((g['fx']-xm)/xs).astype('float32')
        g['yn']=((g['y']-ym)/ys).astype('float32')
    return groups,counts
def crps(s,y):
    m=len(s);return np.abs(s-y).mean(0)-np.abs(s[:,None,...]-s[None,...]).sum((0,1))/(2*m*(m-1))
def recompute(s,y):
    out={k:[] for k in ['crps','daily_mean_crps','ramp_crps','energy','variogram','mae','mse','coverage90','width90']}
    ii,jj=np.triu_indices(24,1);w=1/(jj-ii);w=w/w.sum()
    for a,b in zip(s,y):
        n=len(a);out['crps'].append(crps(a,b).mean())
        out['daily_mean_crps'].append(float(crps(a.mean(-1),b.mean())))
        out['ramp_crps'].append(float(crps(np.abs(np.diff(a,axis=-1)).max(-1),np.abs(np.diff(b)).max())))
        sq=(a*a).sum(1);dist=np.sqrt(np.maximum(sq[:,None]+sq[None,:]-2*a@a.T,0));np.fill_diagonal(dist,0)
        out['energy'].append((np.sqrt(((a-b)**2).sum(1)).mean()-dist.sum()/(2*n*(n-1)))/np.sqrt(24))
        out['variogram'].append(np.sum((np.abs(b[ii]-b[jj])**.5-np.mean(np.abs(a[:,ii]-a[:,jj])**.5,axis=0))**2*w))
        order=np.sort(a,axis=0)
        def quant(p):
            k=p*(n-1);lo=int(np.floor(k));hi=int(np.ceil(k));return order[lo]*(hi-k)+order[hi]*(k-lo) if hi!=lo else order[lo]
        low,high=quant(.05),quant(.95)
        out['mae'].append(np.abs(quant(.5)-b).mean());out['mse'].append(((a.mean(0)-b)**2).mean())
        out['coverage90'].append(((b>=low)&(b<=high)).mean());out['width90'].append((high-low).mean())
    return {k:np.array(v) for k,v in out.items()}
@torch.no_grad()
def replay(model,kind,g,meta,n,rng):
    seed(rng);model.eval();parts=[];qlevels=np.arange(1,20)/20
    for start in range(0,len(g['y']),16):
        h=torch.as_tensor(g['hy'][start:start+16],device='cuda');x=torch.as_tensor(g['hx'][start:start+16],device='cuda');f=torch.as_tensor(g['fx'][start:start+16],device='cuda')
        if kind=='quantile_mlp':
            q=np.sort(model(torch.cat([h.flatten(1),x.flatten(1),f.flatten(1)],dim=1)).cpu().numpy()*meta['scaler']['y_std']+meta['scaler']['y_mean'],axis=-1).astype('float64')
            ext=np.concatenate([(2*q[:,:,0]-q[:,:,1])[:,:,None],q,(2*q[:,:,-1]-q[:,:,-2])[:,:,None]],axis=2)
            arr=np.empty((len(h),n,24));u=(np.arange(n)+.5)/n
            for i in range(len(h)):
                for j in range(24):arr[i,:,j]=np.interp(u,np.r_[0,qlevels,1],ext[i,j])
        else:
            arr=model.inference(h.repeat_interleave(n,0),x.repeat_interleave(n,0),f.repeat_interleave(n,0),num_samples=1).reshape(len(h),n,24).cpu().numpy().astype('float64')
            arr=arr*meta['scaler']['y_std']+meta['scaler']['y_mean']
        parts.append(arr)
    return np.concatenate(parts)
max_score=0.;max_replay=0.;checks=[];fit_count=0;initials={};metric_cells=0
for name in ['wind','solar']:
    meta=json.loads((O/name/'data.json').read_text(encoding='utf-8'));groups,counts=independent_data(name,meta)
    for row in [r for r in result['rows'] if r['dataset']==name]:
        path=O/name/row['fit']/(row['arm']+'.npz')
        with np.load(path) as z:
            s=z['samples'];y=z['targets'];assert np.allclose(y,groups['val']['y'],rtol=0,atol=1e-12)
            assert np.array_equal(z['origins'],groups['val']['origins']);actual=recompute(s,y)
            for key,values in actual.items():
                d=max(float(np.max(np.abs(values-z['score_'+key]))),abs(float(values.mean())-row['summary'][key]));max_score=max(max_score,d)
                assert d<2e-8,(path,key,d);metric_cells+=1
    for sd in [2026371,2026372]:
        schedule=np.load(O/name/f'schedule_{sd}.npy');expected=np.random.default_rng(sd).integers(len(groups['fit']['y']),size=(4000,64),dtype=np.int32)
        assert np.array_equal(schedule,expected)
        for kind in ['quantile_mlp','kite_l1','kite_l2']:
            F=O/name/f'{kind}_{sd}';info=json.loads((F/'fit.json').read_text(encoding='utf-8'));curve=json.loads((F/'curve.json').read_text(encoding='utf-8'))
            selected=min(curve,key=lambda x:x['val_crps']);assert selected['step']==info['selected_step']
            assert sha(F/'selected.pt')==info['checkpoint_sha256'] and sha(O/name/f'schedule_{sd}.npy')==info['schedule_sha256']
            seed(sd);model=make_model(kind,groups['fit']);assert state_sha(model)==info['initial_state_sha256']
            if kind=='kite_l1':initials[(name,sd)]=info['initial_state_sha256']
            if kind=='kite_l2':assert initials[(name,sd)]==info['initial_state_sha256']
            saved=torch.load(F/'selected.pt',map_location='cpu',weights_only=True);assert saved['step']==info['selected_step']
            model.load_state_dict(saved['state']);assert state_sha(model)==info['selected_state_sha256']
            n=128;pred=replay(model,kind,groups['val'],meta,n,sd+90000)
            with np.load(F/('iid.npz' if kind=='quantile_mlp' else 'native.npz')) as z:
                d=float(np.max(np.abs(np.sort(pred,axis=1)-np.sort(z['samples'],axis=1))))
            max_replay=max(max_replay,d);assert d<2e-4,(F,d)
            cp=replay(model,kind,groups['cal'],meta,n,sd+91000)
            with np.load(F/'calibration.npz') as z:
                dc=float(np.max(np.abs(cp-z['samples'])));assert np.allclose(z['targets'],groups['cal']['y'],rtol=0,atol=1e-12)
            max_replay=max(max_replay,dc);assert dc<2e-4
            selection=replay(model,kind,groups['val'],meta,64,sd+80000)
            sc=np.mean([crps(a,b).mean() for a,b in zip(selection,groups['val']['y'])]);assert abs(sc-info['best_selection_val_crps_64'])<2e-5
            assert state_sha(model)==info['selected_state_sha256'];fit_count+=1
            if kind=='quantile_mlp':
                for suffix in ['', '_calibrated']:
                    with np.load(F/('iid'+suffix+'.npz')) as z:reference=np.sort(z['samples'],axis=1)
                    for arm in ['shuffle','analogue']:
                        with np.load(F/(arm+suffix+'.npz')) as z:assert np.array_equal(np.sort(z['samples'],axis=1),reference)
            checks.append({'dataset':name,'kind':kind,'seed':sd,'max_prediction_abs':max(d,dc),'initial_and_selected_hash_verified':True})
            print(json.dumps({'verified_fit':fit_count,'max_score_abs':max_score,'max_prediction_abs':max_replay}),flush=True)
            del model;torch.cuda.empty_cache()
    for mode in ['unconditional','analogue']:
        with np.load(O/name/'hgb'/(mode+'_joint.npz')) as a,np.load(O/name/'hgb'/(mode+'_iid.npz')) as b:
            assert np.array_equal(np.sort(a['samples'],axis=1),np.sort(b['samples'],axis=1))
    def matrix(g):
        f=np.concatenate([g['hy'].reshape(len(g['y']),-1),g['hx'].reshape(len(g['y']),-1),g['fx'].reshape(len(g['y']),-1)],axis=1)
        return np.concatenate([np.repeat(f,24,axis=0),np.tile(np.eye(24,dtype='float32'),(len(f),1))],axis=1)
    est=joblib.load(O/name/'hgb/model.joblib')
    with np.load(O/name/'hgb/point_and_calibration.npz') as z:
        for part,key in [('val','val_mean'),('cal','cal_mean')]:assert np.max(np.abs(est.predict(matrix(groups[part])).reshape(-1,24)-z[key]))<1e-10
out={'passed':True,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'seconds':time.monotonic()-start,
    'neural_fits_verified':fit_count,'hgb_point_models_replayed':2,'score_cells_verified':metric_cells,
    'max_score_abs':max_score,'max_prediction_abs':max_replay,'checks':checks,'test_scored':False,
    'results_sha256':sha(O/'results.json'),'verifier_sha256':sha(pathlib.Path(__file__)),'accepted_research_directions':0}
dest.write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out),flush=True)
