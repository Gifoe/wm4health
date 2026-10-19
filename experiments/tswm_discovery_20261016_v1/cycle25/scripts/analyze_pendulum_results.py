import runtime_profile
import collections,csv,hashlib,json,pathlib,math
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];BASE=ROOT/'outputs/belief_audit';OUT=ROOT/'outputs/analysis';assert not OUT.exists();OUT.mkdir(parents=True)
raw=json.loads((BASE/'results.json').read_text());receipts={}
for name in ['independent_score_verification.json','independent_inference_verification.json']:
    p=BASE/name;r=json.loads(p.read_text());assert r['status']=='pass';assert r.get('raw_result_sha256',r.get('raw_results_sha256'))==hashlib.sha256((BASE/'results.json').read_bytes()).hexdigest();receipts[name]=dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),receipt=r)
groups=collections.defaultdict(list)
for r in raw['rows']:groups[(r['model'],r['n'],r['probe_index'],r['method'])].append(r)
keys=['crps','energy','variogram','mean','variance','squared_error','pit','covariance'];arrays={};aggregates=[];ess=[]
for key,rs in groups.items():
    assert len(rs)==3
    ds=[np.load(BASE/r['path']) for r in rs];a={k:np.stack([d[k] for d in ds]) for k in keys};arrays[key]=a
    mean={k:v.mean(0) for k,v in a.items()};pit=a['pit'];coverage=((pit>=.025)&(pit<=.975)).mean(0)
    agg=dict(model=key[0],n=key[1],probe_index=key[2],method=key[3],crps=float(mean['crps'].mean()),query_crps=mean['crps'].mean(0).tolist(),energy=float(mean['energy'].mean()),variogram=float(mean['variogram'].mean()),rmse=float(np.sqrt(mean['squared_error'].mean())),query_rmse=np.sqrt(mean['squared_error'].mean(0)).tolist(),coverage95=float(coverage.mean()),query_coverage95=coverage.mean(0).tolist(),crps_mc_sd=float(np.std(a['crps'].mean((1,2)),ddof=1)),energy_mc_sd=float(np.std(a['energy'].mean(1),ddof=1)),predictive_variance=float(mean['variance'].mean()))
    aggregates.append(agg)
    if 'ess' in ds[0]:
        e=np.concatenate([d['ess'] for d in ds]);ess.append(dict(model=key[0],n=key[1],probe_index=key[2],method=key[3],minimum=float(e.min()),median=float(np.median(e)),mean=float(e.mean()),fraction_below10=float((e<10).mean()),fraction_below100=float((e<100).mean())))
    for d in ds:d.close()
rng=np.random.default_rng(20262599);idx=rng.integers(0,64,(4000,64));contrasts=[];models=sorted({k[0] for k in groups})
def paired(key,scope,selection):
    model,n,p,method=key;a=arrays[key];b=arrays[(model,n,-1,'no_update')];delta=a['crps'][:,:,selection]-b['crps'][:,:,selection];case=delta.mean((0,2));ci=np.quantile(case[idx].mean(1),[.025,.975])
    da=a['mean'][:,:,selection]-b['mean'][:,:,selection];den=b['variance'][:,:,selection].mean(0).clip(1e-12);squared=da.mean(0)**2-da.var(0,ddof=1)/3
    return dict(model=model,n=n,probe_index=p,method=method,scope=scope,delta_crps=float(case.mean()),bootstrap95=ci.tolist(),delta_energy=float((a['energy']-b['energy']).mean()),delta_coverage95=float((((a['pit'][:,:,selection]>=.025)&(a['pit'][:,:,selection]<=.975)).mean()-((b['pit'][:,:,selection]>=.025)&(b['pit'][:,:,selection]<=.975)).mean())),physical_mean_drift_noise_corrected_squared_prior_sd=float(np.mean(squared/den)),rms_physical_mean_drift_prior_sd=float(np.sqrt(np.maximum(0,np.mean(squared/den)))),query_indices=list(selection),resampling_unit='physical system; average three MC replicates before bootstrapping')
for key in groups:
    if key[2]<0:continue
    contrasts.append(paired(key,'all',[0,1,2,3]))
    if key[1]==0 and key[2]==1:
        contrasts.append(paired(key,'untouched',[0,2,3]));contrasts.append(paired(key,'probed',[1]))
for name,records in [('aggregate_results',aggregates),('paired_contrasts',contrasts),('importance_ess',ess),('raw_scores',raw['rows'])]:
    with (OUT/(name+'.csv')).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in row.items()} for row in records)
summary=dict(status='verified',raw_sha256=hashlib.sha256((BASE/'results.json').read_bytes()).hexdigest(),models=models,systems=64,prediction_rows=len(raw['rows']),aggregate_rows=len(aggregates),contrasts=contrasts,aggregates=aggregates,ess=ess,receipts=receipts,elapsed_seconds=raw['elapsed_seconds'],peak_gpu_bytes=raw['peak_gpu_allocated_bytes'],accepted_final_directions=0,decision='General harmful-update/new-architecture necessity not supported; do not extrapolate the scalar result. Evaluate newer distinct frameworks only if a more specific residual question survives.')
(OUT/'results.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(dict(raw_sha256=summary['raw_sha256'],rows=summary['prediction_rows'],aggregate_rows=len(aggregates),cold_null_native=[r for r in contrasts if r['n']==0 and r['probe_index']==0 and r['method']=='native'],cold_active_native=[r for r in contrasts if r['n']==0 and r['probe_index']==1 and r['method']=='native' and r['scope']=='untouched']),indent=2))
