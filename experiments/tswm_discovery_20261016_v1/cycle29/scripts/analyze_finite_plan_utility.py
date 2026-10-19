"""All predeclared finite-plan decisions; selection never uses true future outputs."""
import csv,json,pathlib,time
import numpy as np
from world_data import ROOT,OUT as DATA,C,sha,dump
EVAL=ROOT/'outputs/world_evaluation_v2';OUT=ROOT/'outputs/finite_plan_utility_v1'
TARGETS=[-1.,0.,1.];HORIZONS=[32,64,128];SEEDS=[2026291,2026292,2026293]

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def writecsv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)));writer.writeheader();writer.writerows(rows)

def main():
    start=time.monotonic();proof=read(ROOT/'outputs/world_verification/model_comparison_v2.json');result=read(EVAL/'results.json')
    assert proof['status']=='passed' and proof['results_sha256']==sha(EVAL/'results.json')
    assert not OUT.exists();OUT.mkdir();rows=[];summary=[];regrets={};input_hashes={}
    for family in ['lti','wh']:
        path=DATA/f'test_{family}.npz';input_hashes[path.name]=sha(path)
        with np.load(path,allow_pickle=False) as f:clean=f['clean'][:,:,C:].astype('float64');u=f['u'][:,:,C:].astype('float64')
        assert clean.shape==u.shape==(128,4,128)
        predictions=[]
        for file in result['predictions']:
            if not file['path'].startswith(family+'_'):continue
            path=EVAL/file['path'];assert sha(path)==file['sha256']
            name=path.stem[len(family)+1:];seed=None
            for s in SEEDS:
                if name.endswith('_'+str(s)):seed=s;name=name[:-(len(str(s))+1)];break
            with np.load(path,allow_pickle=False) as f:pred=f['prediction'].astype('float64')
            assert pred.shape==clean.shape and np.isfinite(pred).all();predictions.append((name,seed,pred))
        predictions += [('always_zero',None,None),('always_factual',None,None)]
        for method,seed,pred in predictions:
            for horizon in HORIZONS:
                action=.01*np.mean(u[:,:,:horizon]**2,axis=2)
                targets=[]
                for target in TARGETS:
                    true_cost=np.mean((clean[:,:,:horizon]-target)**2,axis=2)+action
                    if method=='always_zero':selected=np.ones(128,dtype=int)
                    elif method=='always_factual':selected=np.zeros(128,dtype=int)
                    else:
                        predicted_cost=np.mean((pred[:,:,:horizon]-target)**2,axis=2)+action
                        selected=np.argmin(predicted_cost,axis=1)
                    oracle=np.argmin(true_cost,axis=1);chosen=true_cost[np.arange(128),selected];best=true_cost[np.arange(128),oracle]
                    regret=chosen-best;assert np.isfinite(regret).all() and np.min(regret)>=-1e-10
                    targets.append(regret)
                    summary.append(dict(family=family,method=method,seed=seed,horizon=horizon,target=target,mean_regret=float(regret.mean()),median_regret=float(np.median(regret)),q75_regret=float(np.quantile(regret,.75)),oracle_choice_fraction=float((selected==oracle).mean()),mean_chosen_true_cost=float(chosen.mean()),mean_oracle_true_cost=float(best.mean()),mean_always_zero_true_cost=float(true_cost[:,1].mean()),regret_over_mean_zero_cost=float(regret.mean()/max(true_cost[:,1].mean(),1e-12))))
                    for i in range(128):rows.append(dict(family=family,method=method,seed=seed,horizon=horizon,target=target,system=i,selected_plan=int(selected[i]),oracle_plan=int(oracle[i]),regret=float(regret[i]),chosen_cost=float(chosen[i]),oracle_cost=float(best[i])))
                regrets[(family,method,seed,horizon)]=np.mean(targets,axis=0)
    contrasts=[];rng=np.random.default_rng(202629402)
    for family in ['lti','wh']:
        for horizon in HORIZONS:
            a=np.stack([regrets[(family,'compiled64',s,horizon)] for s in SEEDS]);b=np.stack([regrets[(family,'conditional128',s,horizon)] for s in SEEDS]);delta=a-b;values=[]
            for _ in range(2000):
                seeds=rng.integers(0,3,3);worlds=rng.integers(0,128,128);values.append(float(delta[seeds[:,None],worlds[None,:]].mean()))
            contrasts.append(dict(family=family,horizon=horizon,method_a='compiled64',method_b='conditional128',target_aggregation='equal mean of -1,0,+1 inside world',mean_regret_difference=float(delta.mean()),seed_differences=delta.mean(1).tolist(),difference_p025=float(np.quantile(values,.025)),difference_p975=float(np.quantile(values,.975)),world_win_fraction=float((delta.mean(0)<0).mean()),scope='descriptive crossed bootstrap with3 training seeds'))
    writecsv(OUT/'per_world_decisions.csv',rows);dump(OUT/'summary.json',summary);dump(OUT/'paired_contrasts.json',contrasts)
    result=dict(status='completed',plan_sha256=sha(ROOT/'FINITE_PLAN_UTILITY_PLAN.md'),source_sha256=sha(pathlib.Path(__file__)),verified_predictions_sha256=sha(EVAL/'results.json'),independent_verification_sha256=sha(ROOT/'outputs/world_verification/model_comparison_v2.json'),data_hashes=input_hashes,n_decisions=len(rows),n_summary_rows=len(summary),n_contrasts=len(contrasts),seconds=time.monotonic()-start,accepted_final_directions=0,scope='secondary predeclared finite-four-plan selection;no closed-loop or real-intervention claim')
    dump(OUT/'results.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
