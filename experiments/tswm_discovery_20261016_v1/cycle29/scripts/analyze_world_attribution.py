"""Predeclared all-world attribution and crossed cluster bootstrap; no fitting."""
import csv,json,pathlib,time
import numpy as np
from world_data import ROOT,OUT as DATA,sha,dump,C
EVAL=ROOT/'outputs/world_evaluation_v2';OUT=ROOT/'outputs/world_attribution_v1'
SEEDS=['2026291','2026292','2026293'];HORIZONS=[32,64,128]
CONTRASTS=[('compiled64','conditional128'),('conditional128','conditional96'),('native_attention64_chunk32','native_attention64'),('compiled64_zero_operator','compiled64'),('compiled64_swap_operator','compiled64'),('conditional128_swap_code','conditional128')]

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def writecsv(path,rows):
    with path.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(dict.fromkeys(key for row in rows for key in row)));writer.writeheader();writer.writerows(rows)

def crossed(a,b,rng):
    assert a.shape==b.shape==(3,128)
    bootstrap=[];ratios=[]
    for _ in range(2000):
        seeds=rng.integers(0,3,3);systems=rng.integers(0,128,128)
        aa=a[seeds[:,None],systems[None,:]].mean();bb=b[seeds[:,None],systems[None,:]].mean()
        bootstrap.append(aa-bb);ratios.append(aa/bb if bb>0 else np.nan)
    delta=a-b;perworld=delta.mean(0)
    return dict(mean_difference=float(delta.mean()),ratio_of_means=float(a.mean()/b.mean()),difference_p025=float(np.quantile(bootstrap,.025)),difference_p975=float(np.quantile(bootstrap,.975)),ratio_p025=float(np.quantile(ratios,.025)),ratio_p975=float(np.quantile(ratios,.975)),world_median_difference=float(np.median(perworld)),world_q25_difference=float(np.quantile(perworld,.25)),world_q75_difference=float(np.quantile(perworld,.75)),world_win_fraction=float((perworld<0).mean()),seed_differences=[float(v) for v in delta.mean(1)],n_seeds=3,n_worlds=128,n_bootstrap=2000)

def main():
    proof=read(ROOT/'outputs/world_verification/model_comparison_v2.json');assert proof['status']=='passed' and proof['results_sha256']==sha(EVAL/'results.json')
    assert not OUT.exists();OUT.mkdir();started=time.monotonic()
    with (EVAL/'per_system_metrics.csv').open(encoding='utf-8',newline='') as stream:rows=list(csv.DictReader(stream))
    assert len(rows)==79872
    methods=list(dict.fromkeys(row['method'] for row in rows));arrays={}
    for family in ['lti','wh']:
        for method in methods:
            for horizon in HORIZONS:
                for metric in ['factual','effect']:
                    seeds=SEEDS if method not in ['pretrained_native_icl','arx_context_selected'] else ['']
                    grid=np.full((len(seeds),128),np.nan)
                    for j,seed in enumerate(seeds):
                        grouped={i:[] for i in range(128)}
                        for row in rows:
                            if row['family']!=family or row['method']!=method or row['seed']!=seed or int(row['horizon'])!=horizon:continue
                            use=row['plan']=='factual' if metric=='factual' else row['plan']!='factual'
                            if use:grouped[int(row['system'])].append(float(row['clean_rmse' if metric=='factual' else 'effect_rmse']))
                        expected=1 if metric=='factual' else 3
                        assert all(len(v)==expected for v in grouped.values()),(family,method,horizon,metric,seed)
                        grid[j]=[np.mean(grouped[i]) for i in range(128)]
                    assert np.isfinite(grid).all();arrays[(family,method,horizon,metric)]=grid
    summaries=[];contrasts=[];paired=[];rng=np.random.default_rng(202629401)
    for (family,method,horizon,metric),values in arrays.items():
        means=values.mean(1);summaries.append(dict(family=family,method=method,horizon=horizon,metric=metric,mean=float(means.mean()),seed_std=float(means.std(ddof=1)) if len(means)>1 else None,seed_means=means.tolist(),world_median=float(np.median(values.mean(0))),n_seeds=len(means),n_worlds=128))
    for family in ['lti','wh']:
        for a,b in CONTRASTS:
            for horizon in HORIZONS:
                for metric in ['factual','effect']:
                    aa=arrays[(family,a,horizon,metric)];bb=arrays[(family,b,horizon,metric)]
                    key=dict(family=family,method_a=a,method_b=b,horizon=horizon,metric=metric)
                    contrasts.append(dict(**key,**crossed(aa,bb,rng)))
                    for j,seed in enumerate(SEEDS):
                        paired.extend(dict(**key,seed=seed,system=i,difference=float(aa[j,i]-bb[j,i]),a=float(aa[j,i]),b=float(bb[j,i])) for i in range(128))
    amplitudes=[]
    for family in ['lti','wh']:
        with np.load(DATA/f'test_{family}.npz',allow_pickle=False) as f:clean=f['clean'][:,:,C:]
        for horizon in HORIZONS:
            amplitude=np.sqrt(np.mean((clean[:,1:,:horizon]-clean[:,:1,:horizon])**2,axis=-1)).mean(1)
            amplitudes.append(dict(family=family,horizon=horizon,mean_no_response_rmse=float(amplitude.mean()),median=float(np.median(amplitude)),q25=float(np.quantile(amplitude,.25)),q75=float(np.quantile(amplitude,.75)),n_worlds=len(amplitude)))
    invocation=[]
    for family in ['lti','wh']:
        for seed in SEEDS:
            with np.load(EVAL/f'{family}_native_attention64_{seed}.npz',allow_pickle=False) as f:a=f['prediction'][:,:,:32]
            with np.load(EVAL/f'{family}_native_attention64_chunk32_{seed}.npz',allow_pickle=False) as f:b=f['prediction'][:,:,:32]
            error=float(np.max(np.abs(a-b)));scale=float(np.max(np.abs(a)));assert error<=2e-4*(1+scale),(family,seed,error,scale)
            invocation.append(dict(family=family,seed=seed,h32_prefix_max_abs=error,scale=scale))
    dump(OUT/'summaries.json',summaries);dump(OUT/'paired_contrasts.json',contrasts);dump(OUT/'true_effect_amplitudes.json',amplitudes);dump(OUT/'invocation_checks.json',invocation);writecsv(OUT/'paired_per_world.csv',paired)
    result=dict(status='completed',plan_sha256=sha(ROOT/'RESULT_ATTRIBUTION_PLAN.md'),source_sha256=sha(pathlib.Path(__file__)),verified_results_sha256=sha(EVAL/'results.json'),independent_verification_sha256=sha(ROOT/'outputs/world_verification/model_comparison_v2.json'),n_summaries=len(summaries),n_contrasts=len(contrasts),n_paired_world_rows=len(paired),seconds=time.monotonic()-started,accepted_final_directions=0,scope='all frozen contrasts; crossed bootstrap descriptive with only3 training draws; no new model or test selection')
    dump(OUT/'results.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
