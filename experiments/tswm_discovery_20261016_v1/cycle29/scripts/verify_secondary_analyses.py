"""Independent aggregation and candidate-decision replay from verified raw predictions."""
import csv,json,pathlib,time
from collections import defaultdict
import numpy as np
from world_data import ROOT,OUT as DATA,C,sha,dump
EVAL=ROOT/'outputs/world_evaluation_v2';ATTR=ROOT/'outputs/world_attribution_v1';UTILITY=ROOT/'outputs/finite_plan_utility_v1'

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def csvread(p):
    with p.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def close(a,b,tol=2e-9):
    a=np.asarray(a,dtype='float64');b=np.asarray(b,dtype='float64')
    error=float(np.max(np.abs(a-b)/(1+np.abs(b))));assert np.isfinite(error) and error<tol,(a,b,error)
    return error

def weighted_bootstrap(delta, rng, a=None, b=None):
    vals=[];ratios=[]
    for _ in range(2000):
        sw=np.bincount(rng.integers(0,3,3),minlength=3);ww=np.bincount(rng.integers(0,128,128),minlength=128)
        weights=sw[:,None]*ww[None,:]/384
        vals.append(np.sum(delta*weights))
        if a is not None:ratios.append(np.sum(a*weights)/np.sum(b*weights))
    return np.quantile(vals,[.025,.975]),np.quantile(ratios,[.025,.975]) if a is not None else None

def main():
    started=time.monotonic();proof=read(ROOT/'outputs/world_verification/model_comparison_v2.json');source=read(EVAL/'results.json')
    assert proof['status']=='passed' and proof['results_sha256']==sha(EVAL/'results.json')
    ar=read(ATTR/'results.json');ur=read(UTILITY/'results.json')
    assert ar['verified_results_sha256']==ur['verified_predictions_sha256']==sha(EVAL/'results.json')
    assert source['data_manifest_sha256']==sha(DATA/'manifest.json')
    data_hashes={row['path']:row['sha256'] for row in read(DATA/'manifest.json')['files']}
    for family in ['lti','wh']:
        name=f'test_{family}.npz'
        assert data_hashes[name]==ur['data_hashes'][name]==sha(DATA/name)
    raw=csvread(EVAL/'per_system_metrics.csv');values=defaultdict(list)
    for row in raw:
        metric='factual' if row['plan']=='factual' else 'effect'
        values[(row['family'],row['method'],int(row['horizon']),metric,row['seed'],int(row['system']))].append(float(row['clean_rmse'] if metric=='factual' else row['effect_rmse']))
    arrays={};maxerr=0.
    for summary in read(ATTR/'summaries.json'):
        prefix=(summary['family'],summary['method'],summary['horizon'],summary['metric']);seeds=[''] if summary['n_seeds']==1 else ['2026291','2026292','2026293']
        matrix=np.array([[sum(values[(*prefix,s,i)])/len(values[(*prefix,s,i)]) for i in range(128)] for s in seeds])
        assert np.isfinite(matrix).all();arrays[prefix]=matrix;means=matrix.mean(1)
        maxerr=max(maxerr,close(summary['mean'],means.mean()),close(summary['seed_means'],means),close(summary['world_median'],np.median(matrix.mean(0))))
        if len(seeds)==3:maxerr=max(maxerr,close(summary['seed_std'],means.std(ddof=1)))
    rng=np.random.default_rng(202629401);n_contrasts=0
    for row in read(ATTR/'paired_contrasts.json'):
        key=(row['family'],row['horizon'],row['metric']);a=arrays[(key[0],row['method_a'],key[1],key[2])];b=arrays[(key[0],row['method_b'],key[1],key[2])];d=a-b
        bounds,ratios=weighted_bootstrap(d,rng,a,b)
        maxerr=max(maxerr,close(row['mean_difference'],d.mean()),close(row['seed_differences'],d.mean(1)),close(row['ratio_of_means'],a.mean()/b.mean()),close([row['difference_p025'],row['difference_p975']],bounds),close([row['ratio_p025'],row['ratio_p975']],ratios),close(row['world_win_fraction'],(d.mean(0)<0).mean()))
        n_contrasts+=1
    assert n_contrasts==72
    truths={};inputs={};predictions={}
    for family in ['lti','wh']:
        with np.load(DATA/f'test_{family}.npz',allow_pickle=False) as f:truths[family]=f['clean'][:,:,C:].astype('float64');inputs[family]=f['u'][:,:,C:].astype('float64')
    amplitude_error=0.
    for row in read(ATTR/'true_effect_amplitudes.json'):
        y=truths[row['family']][:,:,:row['horizon']];d=y[:,1:]-y[:,:1]
        amplitude=np.linalg.norm(d,axis=-1).mean(1)/np.sqrt(row['horizon'])
        for name,value in [('mean_no_response_rmse',amplitude.mean()),('median',np.median(amplitude)),('q25',np.quantile(amplitude,.25)),('q75',np.quantile(amplitude,.75))]:
            amplitude_error=max(amplitude_error,close(row[name],value,tol=2e-5))
    for file in source['predictions']:
        path=EVAL/file['path'];assert sha(path)==file['sha256'];family=path.stem.split('_',1)[0];name=path.stem[len(family)+1:];seed=''
        for s in ['2026291','2026292','2026293']:
            if name.endswith('_'+s):name=name[:-(len(s)+1)];seed=s;break
        with np.load(path,allow_pickle=False) as f:predictions[(family,name,seed)]=f['prediction'].astype('float64')
    invocation=read(ATTR/'invocation_checks.json');assert len(invocation)==6
    for row in invocation:
        a=predictions[(row['family'],'native_attention64',row['seed'])][:,:,:32]
        b=predictions[(row['family'],'native_attention64_chunk32',row['seed'])][:,:,:32]
        maxerr=max(maxerr,close(row['h32_prefix_max_abs'],np.max(np.abs(a-b))),close(row['scale'],np.max(np.abs(a))))
    decisions=csvread(UTILITY/'per_world_decisions.csv');groups=defaultdict(list);regrets=defaultdict(dict);selection_ties=0
    for row in decisions:
        family=row['family'];method=row['method'];seed=row['seed'];h=int(row['horizon']);i=int(row['system']);target=float(row['target']);chosen=int(row['selected_plan']);oracle=int(row['oracle_plan'])
        u=inputs[family][i,:,:h];y=truths[family][i,:,:h]
        input_cost=.01*np.einsum('ij,ij->i',u,u)/h
        residual=y-target;cost=np.einsum('ij,ij->i',residual,residual)/h+input_cost
        if method=='always_zero':assert chosen==1
        elif method=='always_factual':assert chosen==0
        else:
            residual=predictions[(family,method,seed)][i,:,:h]-target
            predicted=np.einsum('ij,ij->i',residual,residual)/h+input_cost
            selection_ties+=int(chosen!=np.argmin(predicted))
            maxerr=max(maxerr,close(predicted[chosen],np.min(predicted)))
        maxerr=max(maxerr,close(cost[oracle],np.min(cost)),close(row['chosen_cost'],cost[chosen]),close(row['oracle_cost'],cost[oracle]),close(row['regret'],cost[chosen]-np.min(cost)))
        groups[(family,method,seed,h,target)].append((float(row['regret']),float(row['chosen_cost']),float(row['oracle_cost']),chosen==oracle,float(cost[1])))
        regrets[(family,method,seed,h)].setdefault(i,[]).append(float(row['regret']))
    assert len(decisions)==ur['n_decisions']==64512 and len(groups)==504
    summaries=read(UTILITY/'summary.json');assert len(summaries)==504
    for summary in summaries:
        family=summary['family'];method=summary['method'];seed='' if summary['seed'] is None else str(summary['seed']);h=summary['horizon'];target=summary['target']
        a=np.array(groups[(family,method,seed,h,target)],dtype='float64');assert a.shape==(128,5)
        stats=dict(mean_regret=a[:,0].mean(),median_regret=np.median(a[:,0]),q75_regret=np.quantile(a[:,0],.75),mean_chosen_true_cost=a[:,1].mean(),mean_oracle_true_cost=a[:,2].mean(),oracle_choice_fraction=a[:,3].mean(),mean_always_zero_true_cost=a[:,4].mean(),regret_over_mean_zero_cost=a[:,0].mean()/max(a[:,4].mean(),1e-12))
        for k,v in stats.items():maxerr=max(maxerr,close(summary[k],v))
    rng=np.random.default_rng(202629402);utility_contrasts=read(UTILITY/'paired_contrasts.json')
    for row in utility_contrasts:
        def matrix(method):
            return np.array([[np.mean(regrets[(row['family'],method,s,row['horizon'])][i]) for i in range(128)] for s in ['2026291','2026292','2026293']])
        d=matrix('compiled64')-matrix('conditional128');bounds,_=weighted_bootstrap(d,rng)
        maxerr=max(maxerr,close(row['mean_regret_difference'],d.mean()),close(row['seed_differences'],d.mean(1)),close([row['difference_p025'],row['difference_p975']],bounds),close(row['world_win_fraction'],(d.mean(0)<0).mean()))
    result=dict(status='passed',primary_verification_sha256=sha(ROOT/'outputs/world_verification/model_comparison_v2.json'),attribution_results_sha256=sha(ATTR/'results.json'),utility_results_sha256=sha(UTILITY/'results.json'),source_sha256=sha(pathlib.Path(__file__)),n_attribution_summaries=len(arrays),n_attribution_contrasts=n_contrasts,n_plan_decisions=len(decisions),n_plan_summaries=len(summaries),n_utility_contrasts=len(utility_contrasts),near_tied_decisions=selection_ties,max_relative_numeric_error=maxerr,float32_amplitude_relative_error=amplitude_error,seconds=time.monotonic()-started,scope='all secondary summary and bootstrap contrasts;all finite-plan choices/costs from original prediction arrays;no novelty or broad reliability certificate')
    dump(ROOT/'outputs/secondary_analysis_verification.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
