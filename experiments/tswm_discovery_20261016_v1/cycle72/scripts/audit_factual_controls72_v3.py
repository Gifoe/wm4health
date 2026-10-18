"""All158 native blocks retained; factual-only conditional readouts and native KDE."""
import sys,pathlib,time,contextlib,collections
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common72_v3 import *
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(2);torch.set_default_device('cpu')
assert torch.get_default_device().type=='cpu'
upstream=PROJECT/'upstream'/'epi-cf-benchmark';selection=read(ROOT/'SOURCE_SELECTION72_V1.json')
for r in selection['files']:assert sha(upstream/r['path'])==r['sha256']
sys.path.insert(0,str(upstream/'src'))
from epicf_benchmark import preprocessing as pre
from epicf_benchmark import kde
assert read(OUT/'NATIVE_CONTRACT_QUALIFICATION72_V2.json')['status']=='PINNED_NATIVE_INPUT_CONTRACT_EXECUTED_ON_CONSTRUCTED_SENTINELS'
acq=read(OUT/'DATA_ACQUISITION72_V1.json');begun=time.perf_counter()
settings=[('single','single_policy_data.csv','0-F',['treatment'],pre.SINGLE_SERM_FACTUAL,'infected_sch=0-occ=F'),
          ('multi','multi_policy_data.csv','F-F',['school_treatment','occ_treatment'],pre.MULTI_SERM_FACTUAL,'infected_F-F')]
loaded=[];identity=[]
for setting,file,target,policies,serm,infected in settings:
    spec=next(r for r in acq['files'] if r['file']==file);path=storage('data')/spec['storage_relative'];assert sha(path)==spec['sha256']
    cols=['county','date','day','population_size',target,*policies,*pre.AGE_COLUMNS,*pre.MOBILITY_COLUMNS,*serm,infected]
    assert len(cols)==len(set(cols)) and all('CF' not in c for c in cols)
    df=pd.read_csv(path,usecols=cols,dtype={'county':'string'})
    numeric=[c for c in cols if c not in ('county','date')]
    assert len(df)==158*168 and np.isfinite(df[numeric].to_numpy(dtype=np.float64)).all()
    dates=pd.to_datetime(df['date'],errors='raise');blocks=[]
    for i in range(158):
        p=df.iloc[i*168:(i+1)*168];unit=str(p['county'].iloc[0]);assert (p['county']==unit).all()
        d=dates.iloc[i*168:(i+1)*168].to_numpy(dtype='datetime64[D]')
        assert np.all(np.diff(d)==np.timedelta64(1,'D'))
        assert np.array_equal(p['day'].to_numpy(),np.sort(p['day'].to_numpy()))
        b=p[numeric].to_numpy(dtype=np.float64).tobytes()+d.astype('int64').tobytes()
        blocks.append(dict(instance=i,county=unit,selected_factual_block_sha256=hashlib.sha256(b).hexdigest(),date_start=str(d[0]),date_end=str(d[-1])))
    groups=collections.defaultdict(list)
    for row in blocks:groups[row['county']].append(row)
    duplicate=[]
    for unit,rows in groups.items():
        if len(rows)>1:
            ref=df.iloc[rows[0]['instance']*168:(rows[0]['instance']+1)*168][numeric].to_numpy(dtype=np.float64)
            duplicate.append(dict(county=unit,instances=[r['instance'] for r in rows],selected_factual_blocks_exactly_equal=len({r['selected_factual_block_sha256'] for r in rows})==1,
                max_abs_factual_numeric_difference=max(float(abs(df.iloc[r['instance']*168:(r['instance']+1)*168][numeric].to_numpy(dtype=np.float64)-ref).max()) for r in rows)))
    identity.append(dict(setting=setting,simulation_blocks=158,unique_county_ids=len(groups),rows=len(df),block_steps=168,blocks=blocks,duplicate_county_groups=duplicate,
        source_contiguous_block_order_verified=True,chronological_daily_dates_verified=True,counterfactual_values_parsed=False))
    loaded.append((setting,path,target,policies,serm,infected,df,cols))
dump(OUT/'BLOCK_IDENTITY72_V1.json',dict(utc=utc(),status='NATIVE_BLOCK_IDENTITY_AUDIT_CLOSED_ALL_SOURCE_ROWS_RETAINED',settings=identity,
    source_rows_removed=0,independence_or_duplicate_seed_pairing_assumed=False,counterfactual_values_parsed=False))

def ols(x,y,fit,query):
    mu=x[fit].mean(axis=0);scale=x[fit].std(axis=0);scale=np.where(scale<1e-12,1.,scale)
    design=np.column_stack([np.ones(len(fit)),(x[fit]-mu)/scale])
    coef=np.linalg.lstsq(design,y[fit],rcond=1e-10)[0]
    pred=np.column_stack([np.ones(len(query)),(x[query]-mu)/scale])@coef
    assert np.isfinite(coef).all() and np.isfinite(pred).all();return pred

all_results=[]
for setting,path,target,policies,serm,infected,df,cols in loaded:
    arms=['constant','stock_complement','minus_delta_S','minus_delta_S_E','linear_levels','linear_levels_and_changes']
    predictions={a:[] for a in arms};truth=[];per_block=[];conservation=[]
    fit=np.arange(1,112);query=np.arange(112,168)
    for i in range(158):
        p=df.iloc[i*168:(i+1)*168];unit=str(p['county'].iloc[0]);level=p[serm].to_numpy(dtype=np.float64)
        delta=np.vstack([np.full((1,4),np.nan),np.diff(level,axis=0)]);y=p[target].to_numpy(dtype=np.float64)
        truth.append(y[query]);features=dict(constant=np.empty((168,0)),stock_complement=-level.sum(axis=1,keepdims=True),
            minus_delta_S=-delta[:,0:1],minus_delta_S_E=-delta[:,:2].sum(axis=1,keepdims=True),
            linear_levels=level,linear_levels_and_changes=np.column_stack([level,delta]))
        row=dict(instance=i,county=unit,fit_days=111,diagnostic_days=56,errors={})
        for arm,x in features.items():
            pred=ols(x,y,fit,query);predictions[arm].append(pred);d=pred-y[query]
            row['errors'][arm]=dict(mse=float(np.mean(d*d)),mae=float(np.mean(abs(d))))
        total=level.sum(axis=1)+p[infected].to_numpy(dtype=np.float64)
        conservation.append(dict(instance=i,county=unit,total_initial=float(total[0]),total_max_change=float(abs(total-total[0]).max()),population_initial=float(p['population_size'].iloc[0])))
        per_block.append(row)
    truth=np.stack(truth);predictions={a:np.stack(v) for a,v in predictions.items()};summary=[]
    for arm,pred in predictions.items():
        d=pred-truth;summary.append(dict(arm=arm,raw_mse=float(np.mean(d*d)),raw_mae=float(np.mean(abs(d))),
            median_block_mse=float(np.median([r['errors'][arm]['mse'] for r in per_block])),simulation_blocks=158,unique_county_ids=153,points=158*56,
            information_group='SUPPLIED_FACTUAL_COMPARTMENT_OBSERVATION_DECODING_NOT_FORECAST'))
    array=storage('artifact')/('factual_diagnostic_'+setting+'_v3.npz');assert not array.exists();np.savez_compressed(array,truth=truth,**predictions)
    native_data=storage('data')/('native_train_only_'+setting+'_v3');native_data.mkdir(exist_ok=False)
    y_norm,stats=pre.zscore_outcomes(df[target]);y_norm[0].to_csv(native_data/'y_train.csv',header=False,index=False)
    df[policies].astype(int).to_csv(native_data/'a_train.csv',header=False,index=False)
    base=pre.build_base_covariates(df);comp=df[serm].astype(float).copy();comp.columns=['S','E','R','M']
    pd.concat([base,comp],axis=1).to_csv(native_data/'x_train.csv',index=False);pre.write_stats(native_data,stats)
    model=storage('checkpoint')/('native_kde_'+setting+'_v3');assert not model.exists()
    args=kde.build_train_parser().parse_args(['--data-dir',str(native_data),'--output-dir',str(model),'--policy-dim',str(len(policies)),
        '--num-units','158','--sequence-length','168','--num-covariates','17'])
    check=kde._dataset(native_data,args,'train');assert all(x.device.type=='cpu' for x in check.inputs+check.targets);del check
    log=OUT/('native_kde_'+setting+'_v3_stdout.log')
    with log.open('x',encoding='utf-8') as f,contextlib.redirect_stdout(f):kde.train_kde(args)
    artifacts=[dict(file=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(model.iterdir())]
    record=dict(setting=setting,status='FACTUAL_DIAGNOSTIC_AND_NATIVE_KDE_TRAINING_CLOSED',raw_source_sha256=sha(path),rows=len(df),
        simulation_blocks=158,unique_county_ids=int(df['county'].nunique()),steps_per_block=168,factual_columns_parsed=cols,
        source_row_order_verified=True,counterfactual_target_or_compartment_values_parsed=False,source_rows_removed=0,
        diagnostic_fit_days_per_block=111,diagnostic_days_per_block=56,diagnostic_results=summary,per_block=per_block,compartment_conservation=conservation,
        arrays=dict(storage='artifact_root/cycle72/'+array.name,sha256=sha(array)),
        native_kde=dict(source_body_changed=False,training_only=True,forecast_inference_executed=False,all_native_training_tensors_on_cpu=True,
            bandwidth_grid=args.bandwidth_grid,selected_bandwidth=float((model/'best_bandwidth.txt').read_text()),
            selection_rule='Native in-sample reconstruction includes self neighbors; not validation selection',
            artifact_storage='checkpoint_root/cycle72/'+model.name,artifacts=artifacts,log_sha256=sha(log),
            normalized_training_export_hashes={name:sha(native_data/name) for name in ['y_train.csv','a_train.csv','x_train.csv','outcome_stats.json']}))
    dump(OUT/('FACTUAL_CONTROLS_'+setting.upper()+'72_V3.json'),record);all_results.append(record)
    print(json.dumps(dict(setting=setting,status=record['status'],diagnostic_results=summary,selected_native_training_bandwidth=record['native_kde']['selected_bandwidth'])),flush=True)
result=dict(utc=utc(),status='FACTUAL_ONLY_DIAGNOSTICS_AND_TWO_NATIVE_KDE_FITS_CLOSED',
    settings=[dict(setting=r['setting'],receipt_sha256=sha(OUT/('FACTUAL_CONTROLS_'+r['setting'].upper()+'72_V3.json')),diagnostic_results=r['diagnostic_results']) for r in all_results],
    protocol_sha256=sha(ROOT/'PROTOCOL72_V1.md'),diagnostic_protocol_sha256=sha(ROOT/'FACTUAL_DIAGNOSTIC_PROTOCOL72_V1.md'),
    block_identity_addendum_sha256=sha(ROOT/'BLOCK_IDENTITY_ADDENDUM72_V1.md'),block_identity_audit_sha256=sha(OUT/'BLOCK_IDENTITY72_V1.json'),
    source_audit_sha256=sha(OUT/'SOURCE_AUDIT72_V1.json'),data_acquisition_sha256=sha(OUT/'DATA_ACQUISITION72_V1.json'),
    counterfactual_targets_or_compartments_semantically_accessed=False,counterfactual_scoring_executed=False,
    operational_forecasting_performance_claimed=False,new_method=False,accepted_final_directions=0,wall_seconds=time.perf_counter()-begun,
    runtime=dict(torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,numeric_threads=2,
        cuda_available_observed=bool(torch.cuda.is_available()),cuda_initialized=bool(torch.cuda.is_initialized()),all_native_tensors_cpu=True,affinity=AFFINITY))
dump(OUT/'FACTUAL_CONTROLS72_V3.json',result);print(json.dumps({k:result[k] for k in ['status','counterfactual_scoring_executed','operational_forecasting_performance_claimed','wall_seconds']}),flush=True)
