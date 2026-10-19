"""Fixed paired comparisons, only after independent verification, no test scoring."""
import pathlib,sys,json,hashlib,csv,datetime
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[2]/'cycle29/scripts'))
import runtime_profile
import numpy as np
import pandas as pd
R=pathlib.Path(__file__).resolve().parents[1];O=R/'outputs/learning_v2';A=R/'outputs/analysis_v1'
assert not A.exists(),'Preserve previous analysis'
v=json.loads((R/'outputs/verification_v1.json').read_text(encoding='utf-8'))
res=json.loads((O/'results.json').read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
assert v['passed'] and v['results_sha256']==sha(O/'results.json') and not res['test_scored']
A.mkdir();metrics=['crps','daily_mean_crps','ramp_crps','energy','variogram','mae','mse']
table=[];pair_rows=[];families=[];health=[];covariance={}
seeds=[2026371,2026372];cache={};rng=np.random.default_rng(2026370)
def fetch(data,fit,arm):
    key=(data,fit,arm)
    if key not in cache:
        with np.load(O/data/fit/(arm+'.npz')) as z:
            cache[key]={'origins':z['origins'],**{m:z['score_'+m].copy() for m in metrics}}
    return cache[key]
def bootstrap(a,b,origins):
    t=pd.DatetimeIndex(origins.astype(str));blocks=((t-t.min()).days//7).to_numpy();unique=np.unique(blocks)
    sizes=np.array([(blocks==k).sum() for k in unique]);asum=np.array([a[blocks==k].sum() for k in unique]);bsum=np.array([b[blocks==k].sum() for k in unique])
    weights=rng.multinomial(len(unique),np.full(len(unique),1/len(unique)),size=2000)
    amean=(weights@asum)/(weights@sizes);bmean=(weights@bsum)/(weights@sizes);delta=bmean-amean
    denom=float(b.mean());point=float(b.mean()-a.mean());ci=np.quantile(delta,[.025,.975])
    return {'improvement_abs':point,'ci95_low_abs':float(ci[0]),'ci95_high_abs':float(ci[1]),
        'improvement_pct':100*point/denom if abs(denom)>1e-10 else None,'calendar_blocks':len(unique),'n_origins':len(a)}
def pair(data,label,new,base,seed_label):
    assert np.array_equal(new['origins'],base['origins'])
    for m in metrics:pair_rows.append({'dataset':data,'comparison':label,'seed':seed_label,'metric':m,
        'candidate_mean':float(new[m].mean()),'reference_mean':float(base[m].mean()),**bootstrap(new[m],base[m],new['origins'])})
def mean_cases(items):
    assert all(np.array_equal(x['origins'],items[0]['origins']) for x in items)
    return {'origins':items[0]['origins'],**{m:np.mean([x[m] for x in items],axis=0) for m in metrics}}
for row in res['rows']:
    fit=row['fit'];seed_value='deterministic' if fit=='hgb' else int(fit.rsplit('_',1)[1]);kind=fit if fit=='hgb' else fit.rsplit('_',1)[0]
    table.append({'dataset':row['dataset'],'model':kind,'seed':seed_value,'arm':row['arm'],**row['summary'],
        **{'clipped_'+k:v for k,v in row['clipped_sensitivity'].items()}})
for data in ['wind','solar']:
    by_family={}
    for sd in seeds:
        q=f'quantile_mlp_{sd}';l1=f'kite_l1_{sd}';l2=f'kite_l2_{sd}'
        comparisons=[]
        for suf in ['', '_calibrated']:
            comparisons.extend([(f'quantile_shuffle_vs_iid{suf}',(q,'shuffle'+suf),(q,'iid'+suf)),
                                (f'quantile_analogue_vs_iid{suf}',(q,'analogue'+suf),(q,'iid'+suf)),
                                (f'kite_l2_vs_l1{suf}',(l2,'native'+suf),(l1,'native'+suf))])
        for arm in ['iid','shuffle','analogue']:comparisons.append((f'quantile_{arm}_calibration',(q,arm+'_calibrated'),(q,arm)))
        for fit,tag in [(l1,'l1'),(l2,'l2')]:
            comparisons.append((f'kite_{tag}_calibration',(fit,'native_calibrated'),(fit,'native')))
            comparisons.append((f'kite_{tag}_50_vs_10',(fit,'native_50steps'),(fit,'native')))
        for arm in ['native','native_calibrated']:
            comparisons.append((f'kite_l2_{arm}_vs_quantile_analogue_calibrated',(l2,arm),(q,'analogue_calibrated')))
            comparisons.append((f'kite_l2_{arm}_vs_hgb_analogue_joint',(l2,arm),('hgb','analogue_joint')))
        for label,new,base in comparisons:
            nc,bc=fetch(data,*new),fetch(data,*base);pair(data,label,nc,bc,str(sd))
            by_family.setdefault(label,[]).append((nc,bc))
        for fit in [q,l1,l2]:
            info=json.loads((O/data/fit/'fit.json').read_text(encoding='utf-8'));curve=json.loads((O/data/fit/'curve.json').read_text(encoding='utf-8'))
            health.append({'dataset':data,'fit':fit,**info,'initial_val_crps':curve[0]['val_crps'],
                'relative_selected_gain_from_init':(curve[0]['val_crps']-info['best_selection_val_crps_64'])/max(abs(curve[0]['val_crps']),1e-12)})
    for label,items in by_family.items():pair(data,label,mean_cases([x[0] for x in items]),mean_cases([x[1] for x in items]),'mean_score_over_2_seeds')
    for tag in ['unconditional','analogue']:pair(data,f'hgb_{tag}_joint_vs_iid',fetch(data,'hgb',tag+'_joint'),fetch(data,'hgb',tag+'_iid'),'deterministic')
    with np.load(O/data/'hgb/point_and_calibration.npz') as z:e=z['cal_residual']
    cov=np.cov(e,rowvar=False);eig=np.maximum(np.linalg.eigvalsh(cov),0)[::-1];p=eig/eig.sum();active=e.std(0)>1e-8
    corr=np.corrcoef(e[:,active],rowvar=False);mask=~np.eye(active.sum(),dtype=bool)
    covariance[data]={'label':'CAL target residual, not weather forecast error','n_cal':len(e),'active_leads':int(active.sum()),
        'mean_absolute_offdiagonal_correlation':float(np.abs(corr[mask]).mean()),'top1_covariance_fraction':float(p[0]),
        'top5_covariance_fraction':float(p[:5].sum()),'effective_rank':float(np.exp(-np.sum(p[p>0]*np.log(p[p>0]))))}
    np.save(A/f'{data}_cal_target_residual_cov.npy',cov)
for name,rows in [('all_results.csv',table),('paired_comparisons.csv',pair_rows)]:
    with (A/name).open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
for data in ['wind','solar']:
    for model in ['quantile_mlp','kite_l1','kite_l2','hgb']:
        arms=sorted({r['arm'] for r in table if r['dataset']==data and r['model']==model})
        for arm in arms:
            rows=[r for r in table if r['dataset']==data and r['model']==model and r['arm']==arm]
            families.append({'dataset':data,'model':model,'arm':arm,'n_fits':len(rows),
                **{m:float(np.mean([r[m] for r in rows])) for m in list(rows[0]) if m not in ['dataset','model','seed','arm']},
                **{m+'_seed_std':float(np.std([r[m] for r in rows],ddof=1)) if len(rows)>1 else None for m in metrics}})
out={'status':'completed','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'neural_fits':len(health),
    'health':health,'all_results':table,'family_means':families,'paired_comparisons':pair_rows,'cal_target_residual_structure':covariance,
    'n_selected_initial':sum(h['selected_initial_state'] for h in health),'n_cap_improvement_flags':sum(h['last_1000_improvement_gt_1pct'] for h in health),
    'bootstrap_replicates':2000,'bootstrap_unit':'7 calendar day blocks per dataset/origin; scores averaged over seeds are not forecast ensembles',
    'ci_multiplicity_corrected':False,'test_scored':False,'accepted_research_directions':0,
    'results_sha256':sha(O/'results.json'),'verification_sha256':sha(R/'outputs/verification_v1.json'),
    'analysis_plan_sha256':sha(R/'ANALYSIS_PLAN.md'),'script_sha256':sha(pathlib.Path(__file__))}
(A/'analysis.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps({'status':'completed','result_rows':len(table),'paired_metric_rows':len(pair_rows),
                  'n_cap_flags':out['n_cap_improvement_flags'],'n_selected_initial':out['n_selected_initial'],'test_scored':False}))
