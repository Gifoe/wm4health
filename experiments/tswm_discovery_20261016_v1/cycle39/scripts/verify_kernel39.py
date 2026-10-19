"""Independent pairwise CRPS, raw targets, AR fit/replay and prefix checks."""
import pathlib,sys,json,hashlib,datetime,time
R=pathlib.Path(__file__).resolve().parents[1];B=R.parent;sys.path.insert(0,str(B/'cycle29/scripts'))
import runtime_profile
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
O=R/'outputs/kernel_controls_v1';Q=R/'outputs/native_queries_v2';dest=O/'verification.json';assert not dest.exists()
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
D=json.loads((O/'result.json').read_text(encoding='utf-8'));A=json.loads((Q/'result.json').read_text(encoding='utf-8'));start=time.monotonic()
assert D['source_query_result_sha256']==sha(Q/'result.json') and D['plan_sha256']==sha(R/'ORDINARY_KERNEL_CONTROL_PLAN.md')
assert D['model_state_before']==D['model_state_after']==A['model_state_before'] and D['neural_updates']==0
def pair_score(s,y):
    s=np.asarray(s,dtype=np.float64).reshape(len(s),-1);y=np.asarray(y,dtype=np.float64).reshape(-1);n=len(s)
    return float(np.mean(np.abs(s-y))-pdist(s,metric='cityblock').sum()/(n*(n-1)*s.shape[1]))
max_score=0.;max_ar=0.;cells=0;files=0;prefix_error=0.
for item in D['files']:
    path=O/item['path'];assert sha(path)==item['sha256'];files+=1;task,end_string=path.parent.name.rsplit('_',1);end=int(end_string);rep=int(path.stem.rsplit('_',1)[1])
    raw=B/'cycle34/assets/fev'/('solar_with_weather/15T/train-00000-of-00001.parquet' if task=='solar' else 'jena_weather/10T/train-00000-of-00001.parquet')
    data=pd.read_parquet(raw);field='target' if task=='solar' else 'target_0';series=np.asarray(data.iloc[0][field],dtype=np.float32)
    with np.load(path) as z:
        h=z['history'];truth=z['truth'];assert np.array_equal(h,series[end-672:end]) and np.array_equal(truth,series[end:end+192])
        mu=float(h.mean());sd=float(h.std())+1e-5;target=(truth.astype('float64')-mu)/sd
        prior=np.load(Q/path.parent.name/f'direct_{rep}.npy')[:,:192];prefix_error=max(prefix_error,float(np.max(np.abs(prior-z['native_joint']))))
        for block in [8,96]:prefix_error=max(prefix_error,float(np.max(np.abs(z[f'canonical{block}'][:,:block]-z['native_joint'][:,:block]))))
        for arm in ['native_joint','canonical96','canonical8','ar8']:
            a=(z[arm].astype('float64')-mu)/sd;assert a.shape==(256,192) and np.isfinite(a).all()
            expected={'hourly_crps':pair_score(a,target),'mean_crps':pair_score(a.mean(axis=1),target.mean()),
                      'terminal_crps':pair_score(a[:,-1],target[-1]),'ramp_crps':pair_score(np.abs(np.diff(a,axis=1)).max(axis=1),np.abs(np.diff(target)).max())}
            row=next(x for x in D['rows'] if x.get('dataset')==task and x.get('end')==end and x.get('rep')==rep and x.get('arm')==arm)
            assert row['prediction_seconds']>0
            for key,value in expected.items():
                error=abs(value-row[key]);max_score=max(max_score,error);assert error<1e-8;cells+=1
        # Independent design matrix, explicit coefficient and Gaussian recurrence replay.
        hz=(h-mu)/sd;X=np.array([[1.]+[float(hz[t-j]) for j in range(1,9)] for t in range(8,len(hz))])
        Y=hz[8:];coef=np.linalg.solve(X.T@X+np.diag([0]+[.001]*8),X.T@Y);sigma=float(np.sqrt(np.mean((Y-X@coef)**2)))
        fit=next(x for x in D['ar_context_fits'] if x['dataset']==task and x['end']==end)
        assert np.max(np.abs(coef-np.array(fit['coefficient'])))<1e-8 and abs(sigma-fit['sigma_z'])<1e-9
        index=next(i for i,x in enumerate(A['inputs']) if x['dataset']==task and x['end_index_exclusive']==end)
        rng=np.random.default_rng(rep+index*100);history=np.tile(hz[-8:].astype('float64'),(256,1));draws=[]
        for t in range(192):
            point=coef[0]+sum(coef[j]*history[:,-j] for j in range(1,9))+sigma*rng.normal(size=256)
            draws.append(point);history=np.concatenate([history[:,1:],point[:,None]],axis=1)
        replay=np.array(draws).T*sd+mu;max_ar=max(max_ar,float(np.max(np.abs(replay-z['ar8']))));assert max_ar<1e-6
assert files==16 and cells==256 and prefix_error<2e-4 and len(D['ar_context_fits'])==8
out={'passed':True,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'seconds':time.monotonic()-start,
     'prediction_files_sha_verified':files,'score_cells_verified':cells,'max_score_abs':max_score,'max_ar_replay_abs':max_ar,
     'max_native_first_prefix_abs':prefix_error,'ar_history_fits':8,'neural_updates':0,'test_scored':False,
     'results_sha256':sha(O/'result.json'),'verifier_sha256':sha(pathlib.Path(__file__))}
dest.write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out))
