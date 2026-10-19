"""Independent raw-input and saved-array statistical verification; no model fitting."""
import pathlib,sys,json,hashlib,datetime,math,time
R=pathlib.Path(__file__).resolve().parents[1];B=R.parent;sys.path.insert(0,str(B/'cycle29/scripts'))
import runtime_profile
import numpy as np
import pandas as pd
O=R/'outputs/native_queries_v2';dest=O/'verification.json';assert not dest.exists()
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
D=json.loads((O/'result.json').read_text(encoding='utf-8'));start=time.monotonic()
assert D['status']=='completed_pending_independent_verification' and D['new_training_updates']==0 and not D['test_scored']
assert D['model_state_before']==D['model_state_after'] and D['splits']==[96,352,360]
assert D['plan_sha256']==sha(R/'NATIVE_QUERY_PLAN_V2.md')
assert max(x['relative_max'] for x in D['qualification'])<2e-5
source=json.loads((B/'cycle34/upstream_source/Sundial/source_manifest.json').read_text(encoding='utf-8'))
for item in source['files']:assert sha(B/'cycle34/upstream_source/Sundial'/item['path'])==item['sha256']
assert sha(B/'cycle34/assets/Sundial/model.safetensors')=='414435b508391f92afadd2aaeec418c806776aeccbce12e638d73a139ca5ca78'
def independent_probes(u,v,mean,scale):
    out=np.empty((len(u),9))
    for i in range(len(u)):
        prefix=(u[i].astype('float64')-mean)/scale
        weights=[1.,math.tanh(float(np.sum(prefix)/len(prefix))),math.tanh(float(prefix[-1]))]
        suffix=(v[i].astype('float64')-mean)/scale
        # Explicit average of nonlinear per-inner statistics.
        features=np.zeros(3)
        for draw in suffix:
            features+=np.array([np.sum(draw)/len(draw),draw[-1],np.dot(draw,draw)/len(draw)])
        features/=len(suffix)
        out[i]=[w*g for w in weights for g in features]
    return out
max_probe=0.;max_stat=0.;checked_rows=0;input_count=0;array_files=0
for rec in D['inputs']:
    if rec['excluded'] is not None:continue
    task=rec['dataset'];end=rec['end_index_exclusive'];folder=O/f'{task}_{end}'
    path=B/'cycle34/assets/fev'/('solar_with_weather/15T/train-00000-of-00001.parquet' if task=='solar' else 'jena_weather/10T/train-00000-of-00001.parquet')
    assert sha(path)==rec['raw_sha256'];data=pd.read_parquet(path);assert len(data)==1
    expected=np.asarray(data.iloc[0][rec['field']],dtype=np.float32)[end-672:end]
    actual=np.load(folder/'history.npy');assert np.array_equal(expected,actual) and np.isfinite(actual).all()
    anchor=864 if task=='solar' else 0;assert end in [anchor+672+x for x in [0,336,672,1008]]
    info=json.loads((folder/'input.json').read_text(encoding='utf-8'));assert sha(folder/'history.npy')==info['history_sha256']
    mean=info['history_mean'];scale=info['history_scale'];assert abs(mean-float(actual.mean()))<1e-4 and scale>0
    input_count+=1
    for rep in [2026391,2026392]:
        full=np.load(folder/f'direct_{rep}.npy');assert full.shape==(256,720) and np.isfinite(full).all()
        for split in [96,352,360]:
            p=folder/f'query_{rep}_{split}.npz'
            receipt=next(x for x in D['prediction_files'] if x['path']==p.relative_to(O).as_posix());assert sha(p)==receipt['sha256'];array_files+=1
            with np.load(p) as z:
                assert np.array_equal(z['prefix'],full[:,:split]) and np.array_equal(z['direct_suffix'],full[:,split:2*split])
                a=independent_probes(z['prefix'],z['direct_suffix'][:,None,:],mean,scale)
                max_probe=max(max_probe,float(np.max(np.abs(a-z['direct_probes']))))
                for mode in ['native','fixed_reference']:
                    pred=z[mode+'_suffix'];assert pred.shape==(256,8,split) and np.isfinite(pred).all()
                    c=independent_probes(z['prefix'],pred,mean,scale);max_probe=max(max_probe,float(np.max(np.abs(c-z[mode+'_probes']))))
                    selected=[r for r in D['rows'] if r['dataset']==task and r['end_index_exclusive']==end and r['rep']==rep and r['split']==split and r['mode']==mode]
                    assert len(selected)==9
                    for row in selected:
                        j=row['probe'];delta=a[:,j]-c[:,j];avg=float(delta.mean());se=float(np.sqrt(np.sum((delta-avg)**2)/(len(delta)-1)/len(delta)))
                        expected_fields={'difference':avg,'mc_se_outer':se,'mc95_low':avg-1.96*se,'mc95_high':avg+1.96*se,
                                         'direct':float(a[:,j].mean()),'composed':float(c[:,j].mean())}
                        for key,val in expected_fields.items():max_stat=max(max_stat,abs(val-row[key]));assert abs(val-row[key])<1e-8
                        checked_rows+=1
with np.load(O/'statistic_fixtures.npz') as z:
    a=independent_probes(z['prefix'],z['direct'],0.,1.);good=independent_probes(z['prefix'],z['coherent'],0.,1.);bad=independent_probes(z['prefix'],z['independent'],0.,1.)
    for c in [good,bad]:
        assert np.all(np.isfinite(c))
    delta=a-good;zgood=delta.mean(0)/(delta.std(0,ddof=1)/np.sqrt(len(a)));assert np.max(np.abs(zgood))<5
    delta=a-bad;zbad=delta.mean(0)/(delta.std(0,ddof=1)/np.sqrt(len(a)));assert abs(zbad[3])>10
assert input_count==8 and array_files==48 and checked_rows==len(D['rows'])==864
assert max_probe<1e-8 and max_stat<1e-8
out={'passed':True,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'seconds':time.monotonic()-start,
     'raw_contexts_verified':input_count,'prediction_npz_sha_verified':array_files,'moment_rows_verified':checked_rows,
     'max_probe_abs':max_probe,'max_summary_abs':max_stat,'statistic_control_passed':True,
     'scope':'Independent raw history, all saved samples/probes/MC statistics and source/weight SHA; native API equivalence is worker qualification, not independently rerun here',
     'test_scored':False,'new_training_updates':0,'accepted_directions':0,'results_sha256':sha(O/'result.json'),'verifier_sha256':sha(pathlib.Path(__file__))}
dest.write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out),flush=True)
