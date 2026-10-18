"""Factual-only conditional readouts plus exact native KDE training, CPU only."""
import sys,pathlib,csv,io,contextlib,ast,collections,time
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common72_v2 import *
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(2)
assert not torch.cuda.is_available()
upstream=PROJECT/'upstream'/'epi-cf-benchmark'
selection=read(ROOT/'SOURCE_SELECTION72_V1.json')
for r in selection['files']:assert sha(upstream/r['path'])==r['sha256']
sys.path.insert(0,str(upstream/'src'))
from epicf_benchmark import preprocessing as pre
from epicf_benchmark import kde
assert read(OUT/'NATIVE_CONTRACT_QUALIFICATION72_V2.json')['status']=='PINNED_NATIVE_INPUT_CONTRACT_EXECUTED_ON_CONSTRUCTED_SENTINELS'
acq=read(OUT/'DATA_ACQUISITION72_V1.json')
assert acq['status']=='OFFICIAL_SMALL_DATA_FILES_PINNED_HEADERS_ONLY'

def standard_ols(x,y,fit,query):
    mu=x[fit].mean(axis=0);scale=x[fit].std(axis=0);scale=np.where(scale<1e-12,1.,scale)
    a=np.column_stack([np.ones(len(fit)),(x[fit]-mu)/scale])
    b=np.linalg.lstsq(a,y[fit],rcond=1e-10)[0]
    pred=np.column_stack([np.ones(len(query)),(x[query]-mu)/scale])@b
    assert np.isfinite(b).all() and np.isfinite(pred).all()
    return pred

all_results=[];begun=time.perf_counter()
for setting,file,target_col,policy_cols,serm,infected in [
    ('single','single_policy_data.csv','0-F',['treatment'],pre.SINGLE_SERM_FACTUAL,'infected_sch=0-occ=F'),
    ('multi','multi_policy_data.csv','F-F',['school_treatment','occ_treatment'],pre.MULTI_SERM_FACTUAL,'infected_F-F')]:
    spec=next(x for x in acq['files'] if x['file']==file);path=storage('data')/spec['storage_relative']
    assert sha(path)==spec['sha256']
    columns=['county','date','day','population_size',target_col,*policy_cols,*pre.AGE_COLUMNS,*pre.MOBILITY_COLUMNS,*serm,infected]
    assert len(columns)==len(set(columns))
    assert all('CF' not in x for x in columns)
    df=pd.read_csv(path,usecols=columns,dtype={'county':'string'})
    numeric=[c for c in columns if c not in ('county','date')]
    assert np.isfinite(df[numeric].to_numpy(dtype=np.float64)).all()
    units=df['county'].drop_duplicates().tolist();assert len(units)==158 and len(df)==158*168
    dates=pd.to_datetime(df['date'],errors='raise')
    counts=df.groupby('county',sort=False).size();assert (counts==168).all()
    arms=['constant','stock_complement','minus_delta_S','minus_delta_S_E','linear_levels','linear_levels_and_changes']
    predictions={arm:[] for arm in arms};truth=[];per_unit=[];conservation=[]
    fit=np.arange(1,112);query=np.arange(112,168)
    for i,unit in enumerate(units):
        piece=df.iloc[i*168:(i+1)*168]
        assert (piece['county']==unit).all(),'Source county-by-time reshape order is not established'
        td=dates.iloc[i*168:(i+1)*168]
        assert np.all(np.diff(td.to_numpy(dtype='datetime64[D]'))==np.timedelta64(1,'D'))
        assert np.array_equal(piece['day'].to_numpy(),np.sort(piece['day'].to_numpy()))
        level=piece[serm].to_numpy(dtype=np.float64);delta=np.vstack([np.full((1,4),np.nan),np.diff(level,axis=0)])
        y=piece[target_col].to_numpy(dtype=np.float64);truth.append(y[query])
        features=dict(constant=np.empty((168,0)),stock_complement=-level.sum(axis=1,keepdims=True),
            minus_delta_S=-delta[:,0:1],minus_delta_S_E=-delta[:,:2].sum(axis=1,keepdims=True),
            linear_levels=level,linear_levels_and_changes=np.column_stack([level,delta]))
        row=dict(county=str(unit),fit_days=111,diagnostic_days=56,errors={})
        for arm,x in features.items():
            p=standard_ols(x,y,fit,query);predictions[arm].append(p);d=p-y[query]
            row['errors'][arm]=dict(mse=float(np.mean(d*d)),mae=float(np.mean(abs(d))))
        total=level.sum(axis=1)+piece[infected].to_numpy(dtype=np.float64)
        conservation.append(dict(county=str(unit),total_initial=float(total[0]),total_max_change=float(abs(total-total[0]).max()),
            population_size_initial=float(piece['population_size'].iloc[0]),population_vs_compartment_total=float(piece['population_size'].iloc[0]-total[0])))
        per_unit.append(row)
    truth=np.stack(truth);predictions={arm:np.stack(p) for arm,p in predictions.items()}
    summaries=[]
    for arm,p in predictions.items():
        d=p-truth
        summaries.append(dict(arm=arm,raw_mse=float(np.mean(d*d)),raw_mae=float(np.mean(abs(d))),
            median_county_mse=float(np.median([r['errors'][arm]['mse'] for r in per_unit])),
            counties=158,points=158*56,information_group='SUPPLIED_FACTUAL_COMPARTMENT_OBSERVATION_DECODING_NOT_FORECAST'))
    array=storage('artifact')/('factual_diagnostic_'+setting+'_v1.npz');assert not array.exists()
    np.savez_compressed(array,truth=truth,**predictions)
    native_data=storage('data')/('native_train_only_'+setting+'_v1');native_data.mkdir(exist_ok=False)
    y_norm,stats=pre.zscore_outcomes(df[target_col])
    y_norm[0].to_csv(native_data/'y_train.csv',header=False,index=False)
    df[policy_cols].astype(int).to_csv(native_data/'a_train.csv',header=False,index=False)
    base=pre.build_base_covariates(df);compartments=df[serm].astype(float).copy();compartments.columns=['S','E','R','M']
    pd.concat([base,compartments],axis=1).to_csv(native_data/'x_train.csv',index=False)
    pre.write_stats(native_data,stats)
    model=storage('checkpoint')/('native_kde_'+setting+'_v1');assert not model.exists()
    args=kde.build_train_parser().parse_args(['--data-dir',str(native_data),'--output-dir',str(model),
        '--policy-dim',str(len(policy_cols)),'--num-units','158','--sequence-length','168','--num-covariates','17'])
    log=OUT/('native_kde_'+setting+'_v1_stdout.log')
    with log.open('x',encoding='utf-8') as f,contextlib.redirect_stdout(f):kde.train_kde(args)
    artifacts=[]
    for p in sorted(model.iterdir()):
        assert p.is_file();artifacts.append(dict(file=p.name,bytes=p.stat().st_size,sha256=sha(p)))
    record=dict(setting=setting,status='FACTUAL_DIAGNOSTIC_AND_NATIVE_KDE_TRAINING_CLOSED',
        raw_source_sha256=sha(path),rows=len(df),counties=len(units),steps_per_county=168,factual_columns_parsed=columns,
        counterfactual_target_or_compartment_values_parsed=False,source_row_order_verified=True,
        diagnostic_fit_days_per_county=111,diagnostic_days_per_county=56,diagnostic_results=summaries,
        per_county=per_unit,compartment_conservation=conservation,
        arrays=dict(storage='artifact_root/cycle72/'+array.name,sha256=sha(array)),
        native_kde=dict(source_body_changed=False,training_only=True,forecast_inference_executed=False,
            bandwidth_grid=args.bandwidth_grid,selected_bandwidth=float((model/'best_bandwidth.txt').read_text()),
            selection_rule='Native in-sample training reconstruction, includes self neighbors; not validation selection',
            artifact_storage='checkpoint_root/cycle72/'+model.name,artifacts=artifacts,log_sha256=sha(log),
            normalized_training_export_hashes={name:sha(native_data/name) for name in ['y_train.csv','a_train.csv','x_train.csv','outcome_stats.json']}))
    dump(OUT/('FACTUAL_CONTROLS_'+setting.upper()+'72_V1.json'),record);all_results.append(record)
    print(json.dumps(dict(setting=setting,status=record['status'],diagnostic_results=summaries,
        selected_native_training_bandwidth=record['native_kde']['selected_bandwidth'])),flush=True)

result=dict(utc=utc(),status='FACTUAL_ONLY_INPUT_CONTRACT_DIAGNOSTICS_AND_TWO_NATIVE_KDE_FITS_CLOSED',
    settings=[dict(setting=r['setting'],receipt_sha256=sha(OUT/('FACTUAL_CONTROLS_'+r['setting'].upper()+'72_V1.json')),
        diagnostic_results=r['diagnostic_results']) for r in all_results],
    protocol_sha256=sha(ROOT/'PROTOCOL72_V1.md'),diagnostic_protocol_sha256=sha(ROOT/'FACTUAL_DIAGNOSTIC_PROTOCOL72_V1.md'),
    source_audit_sha256=sha(OUT/'SOURCE_AUDIT72_V1.json'),data_acquisition_sha256=sha(OUT/'DATA_ACQUISITION72_V1.json'),
    counterfactual_targets_or_compartments_semantically_accessed=False,counterfactual_scoring_executed=False,
    operational_forecasting_performance_claimed=False,new_method=False,accepted_final_directions=0,
    wall_seconds=time.perf_counter()-begun,runtime=dict(torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,numeric_threads=2,cuda_visible=False,affinity=AFFINITY))
dump(OUT/'FACTUAL_CONTROLS72_V1.json',result)
print(json.dumps({k:result[k] for k in ['status','counterfactual_scoring_executed','operational_forecasting_performance_claimed','wall_seconds']}),flush=True)
