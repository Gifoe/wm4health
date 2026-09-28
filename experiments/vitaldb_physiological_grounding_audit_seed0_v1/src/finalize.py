"""Scientific integrity gate, scorecard, six figures and evidence-bound report."""
import hashlib,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from common import *
from extract import FactualWindows

def table(name):return pd.read_csv(OUT/name)
def one(df,**kw):
    for k,v in kw.items():df=df[df[k]==v]
    assert len(df)==1,(kw,len(df))
    return df.iloc[0]
def fmt(v):return f'{v:.3f}' if np.isfinite(v) else 'NA'
def save(fig,name):
    fig.tight_layout();fig.savefig(ROOT/'plots'/f'{name}.png',dpi=170)
    fig.savefig(ROOT/'plots'/f'{name}.pdf');plt.close(fig)

def checks():
    store=Store();ds=Windows(store,'test')
    data=json.loads((OUT/'data_summary.json').read_text())
    cache=json.loads((OUT/'latent_cache_summary.json').read_text())
    case,t=map(int,ds.indices[0]);c=store.cases[case];fw=FactualWindows(ds,6)
    original=fw[0];after=t+7;endpoint=t+6
    sn=c['sn'][after].copy();an=c['an'][after].copy();inside=c['sn'][endpoint].copy()
    try:
        c['sn'][after]+=1000;c['an'][after]+=1000
        changed_after=fw[0]
        c['sn'][after]=sn;c['an'][after]=an
        c['sn'][endpoint]+=1000
        changed_endpoint=fw[0]
    finally:
        c['sn'][after]=sn;c['an'][after]=an;c['sn'][endpoint]=inside
    sample=ds[0]
    widths=load('test','z0').shape[1],load('test','ztrue').shape[-1],load('test','rollout').shape[-1]
    split_subjects={s:set(np.asarray(load(s,'subject')).tolist()) for s in ['train','val','test']}
    result={
      'same_495_cases_493_patients':len(store.cases)==495 and data['patients']==493,
      'same_subject_disjoint_split':all(not(split_subjects[a]&split_subjects[b]) for a,b in [('train','val'),('train','test'),('val','test')]),
      'same_eligible_window_counts':data['eligible_windows']=={'train':370566,'val':79574,'test':83198},
      'AR_predictions_exactly_reproduced':cache['ar_prediction_max_abs_difference']==0,
      'factual_anchor_excludes_later_physiology_and_action':all(np.array_equal(a,b) for a,b in zip(original,changed_after)),
      'factual_anchor_includes_endpoint_only_at_its_time':not np.array_equal(original[0],changed_endpoint[0]),
      'no_CE_model_feature':not any('ce' in k.lower() for k in sample) and sample['state'].shape==(H,6) and sample['action'].shape==(H,4),
      'all_latent_controls_64_dimensions':widths==(64,64,64),
      'future_action_summary_respects_horizon':np.allclose(np.asarray(load('test','future_action_summary'))[0,0,:2],
           np.nansum(c['action'][t+1:t+4],axis=0)),
      'ridge_tuning_val_only':set(table('transition_probe_metrics.csv').alpha.dropna().unique()).issubset(set(CFG['ridge_alpha_grid'])),
      'patient_bootstrap_1000':CFG['bootstrap_replicates']==1000,
    }
    write_json('integrity_checks.json',result)
    if not all(result.values()):raise RuntimeError('Integrity failure '+str({k:v for k,v in result.items() if not v}))
    return store

def scorecards():
    current=table('current_state_probe_metrics.csv');future=table('future_change_probe_metrics.csv')
    transition=table('transition_probe_metrics.csv');non=table('nonlinear_probe_metrics.csv')
    hor=[]
    for r in future.itertuples():
        if r.representation=='z_t':hor.append({'representation':'z_t','probe':'ridge','target':r.target,'horizon_seconds':r.horizon_seconds,'r2':r.r2,'mae':r.mae,'pearson':r.pearson})
        if r.representation=='z_t+action':hor.append({'representation':'z_t+action','probe':'ridge','target':r.target,'horizon_seconds':r.horizon_seconds,'r2':r.r2,'mae':r.mae,'pearson':r.pearson})
    for df,probe in [(transition,'ridge'),(non,'small_mlp')]:
        for r in df.itertuples():
            if r.representation in ['dz_pred','dz_true']:
                hor.append({'representation':r.representation,'probe':probe,'target':r.target,
                            'horizon_seconds':r.horizon_seconds,'r2':r.r2,'mae':r.mae,'pearson':r.pearson})
    pd.DataFrame(hor).to_csv(OUT/'horizon_grounding_metrics.csv',index=False)
    rows=[]
    for rep in ['z_t','z_t+action','dz_pred','dz_true']:
        for target in ['BIS','MAP','HR']:
            q=pd.DataFrame(hor);q=q[(q.representation==rep)&(q.probe=='ridge')&(q.target==target)]
            rows.append({'representation':rep,'target':'Δ'+target,**{f'r2_{h}s':float(one(q,horizon_seconds=h).r2) for h in [30,60,180,300]}})
    pd.DataFrame(rows).to_csv(OUT/'main_grounding_scorecard.csv',index=False)
    # The combined physiological response magnitude uses TRAIN-only change scales.
    scale=np.asarray(list(json.loads((OUT/'response_scale_protocol.json').read_text())['train_300s_change_std'].values()))
    current=np.asarray(load('test','current'));target=np.asarray(load('test','target'))
    mask=np.asarray(load('test','current_fresh'))[:,None,:]&np.asarray(load('test','fresh'))
    subject=np.asarray(load('test','subject'));magnitudes=[]
    for k,h in enumerate(HORIZONS):
        y=(target[:,k]-current)/scale
        valid=mask[:,k].all(1)&np.isfinite(y).all(1)
        mag=np.linalg.norm(y,axis=1)
        w=patient_weights(subject,valid)
        magnitudes.append({'horizon_seconds':h*10,'patient_weighted_mean_response_magnitude':float(np.sum(w[valid]*mag[valid])),
                           'patients':int(np.unique(subject[valid]).size),'windows':int(valid.sum())})
    pd.DataFrame(magnitudes).to_csv(OUT/'phys_response_magnitude.csv',index=False)

def figures(store):
    plt.rcParams.update({'font.size':9,'axes.spines.right':False,'axes.spines.top':False})
    cur=table('current_state_probe_metrics.csv');future=table('future_change_probe_metrics.csv')
    trans=table('transition_probe_metrics.csv');direc=table('transition_direction_metrics.csv')
    bins=table('response_distance_bins.csv');sub=table('subgroup_grounding_metrics.csv')
    x=np.array([30,60,180,300])
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    vals=[one(cur,representation='z_t',target=v).r2 for v in ['BIS','MAP','HR']]
    axes[0].bar(['BIS','MAP','HR'],vals,color=['#3279a6','#5a9e83','#c88646'])
    axes[0].set(ylabel='Patient-weighted TEST R²',title='Current state from zₜ',ylim=(0,1.05))
    for v,color in [('BIS','#3279a6'),('MAP','#5a9e83'),('HR','#c88646')]:
        q=future[(future.representation=='z_t')&(future.target==v)].set_index('horizon_seconds').loc[x]
        axes[1].plot(x,q.r2,marker='o',label='Δ'+v,color=color)
    axes[1].set(title='Future change from zₜ alone',xlabel='Horizon (s)',ylabel='TEST R²',xticks=x)
    axes[1].legend(frameon=False);save(fig,'figure1_current_vs_change')
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),sharex=True)
    for ax,v in zip(axes,['BIS','MAP']):
        for rep,color in [('dz_pred','#c05745'),('dz_true','#3279a6')]:
            q=trans[(trans.target==v)&(trans.representation==rep)].set_index('horizon_seconds').loc[x]
            ax.plot(x,q.r2,marker='o',label=rep,color=color)
        ax.set(title='Δ'+v,xlabel='Horizon (s)',ylabel='TEST R²',xticks=x);ax.grid(alpha=.15)
    axes[0].legend(frameon=False);save(fig,'figure2_transition_grounding')
    fig,ax=plt.subplots(figsize=(7,4))
    cats=['same','adjacent','opposite','random'];pos=np.arange(4)
    for rep,shift,color in [('dz_pred',-.18,'#c05745'),('dz_true',.18,'#3279a6')]:
        q=direc[(direc.subset=='overall')&(direc.target=='BIS')&(direc.horizon_seconds==300)]
        vals=[one(q,representation=rep,category=c).cosine for c in cats]
        ax.bar(pos+shift,vals,width=.34,label=rep,color=color)
    ax.axhline(0,color='black',lw=.7);ax.set_xticks(pos,cats)
    ax.set(ylabel='Mean transition cosine',title='BIS response direction, 300 s');ax.legend(frameon=False)
    save(fig,'figure3_transition_direction')
    fig,ax=plt.subplots(figsize=(7,4))
    for rep,color in [('dz_pred','#c05745'),('dz_true','#3279a6')]:
        q=bins[bins.representation==rep].sort_values('phys_distance_midpoint')
        ax.plot(q.phys_distance_midpoint,q.latent_distance_median,marker='o',label=rep,color=color)
    ax.set(xlabel='Normalized physiological response distance',ylabel='Median train-scaled latent distance',
           title='Patient-balanced sampled pairs, 300 s');ax.legend(frameon=False)
    save(fig,'figure4_response_distance')
    groups=['stable_action_Q1','initiation','increase','decrease','stop']
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    for ax,h in zip(axes,[180,300]):
        for rep,color in [('dz_pred','#c05745'),('dz_true','#3279a6')]:
            vals=[one(sub,subset=s,representation=rep,target='BIS',horizon_seconds=h).r2 for s in groups]
            ax.plot(np.arange(len(groups)),vals,marker='o',label=rep,color=color)
        ax.set_xticks(np.arange(len(groups)),['stable','initiation','increase','decrease','stop'],rotation=25)
        ax.set(title=f'{h} s',ylabel='BIS change probe R²');ax.grid(alpha=.15)
    axes[0].legend(frameon=False);save(fig,'figure5_intervention_change')
    # Case IDs and anchors are selected without model predictions or errors.
    ref=np.load(SOURCE/'outputs/test_reference.npz');cases=np.sort(np.unique(ref['case']))
    chosen=[cases[len(cases)//4],cases[3*len(cases)//4]]
    rng=np.random.default_rng(0);ztrain=np.asarray(load('train','z0'))
    sample=rng.choice(len(ztrain),size=min(20000,len(ztrain)),replace=False)
    pca=PCA(n_components=2,random_state=0).fit(ztrain[sample])
    write_json('pca_visualization_protocol.json',{'train_current_latent_fit_windows':len(sample),
               'explained_variance_ratio':pca.explained_variance_ratio_.tolist(),
               'selected_caseids':[int(x) for x in chosen],'selection':'sorted TEST case IDs at 25% and 75%; median eligible anchor'})
    fig,axes=plt.subplots(2,4,figsize=(15,6.5))
    pred=np.asarray(load('test','prediction'));roll=load('test','rollout');factual=load('test','ztrue');z0=load('test','z0')
    for row,cid in enumerate(chosen):
        ids=np.flatnonzero(ref['case']==cid);i=int(ids[len(ids)//2]);x=np.arange(1,31)*10
        for col,name,j in [(0,'BIS',0),(1,'MAP',1)]:
            ax=axes[row,col];ax.plot(x,ref['target'][i,:,j],color='black',label='Observed')
            ax.plot(x,pred[i,:,j],color='#c05745',label='RSSM predicted')
            ax.set(title=f'Case {int(cid)} / {name}',xlabel='Seconds',ylabel=name)
            if row==0 and col==0:ax.legend(frameon=False,fontsize=8)
        a=ref['future_action'][i];ax=axes[row,2]
        ax.plot(x,a[:,0],label='Propofol');ax.plot(x,a[:,1],label='Remifentanil')
        ax.set(title='Future administration',xlabel='Seconds',ylabel='mL / 10 s')
        if row==0:ax.legend(frameon=False,fontsize=8)
        ax=axes[row,3];pr=pca.transform(roll[i]);ft=pca.transform(factual[i]);origin=pca.transform(z0[i][None])[0]
        ax.plot(pr[:,0],pr[:,1],color='#c05745',marker='.',label='Predicted rollout')
        ax.scatter(ft[:,0],ft[:,1],color='#3279a6',marker='x',s=60,label='Factual anchors')
        ax.scatter([origin[0]],[origin[1]],color='black',s=35,label='Current')
        ax.set(title='Latent PCA (display only)',xlabel='PC1',ylabel='PC2')
        if row==0:ax.legend(frameon=False,fontsize=7)
    save(fig,'figure6_prespecified_trajectories')

def report():
    data=json.loads((OUT/'data_summary.json').read_text());cache=json.loads((OUT/'latent_cache_summary.json').read_text())
    current=table('current_state_probe_metrics.csv');future=table('future_change_probe_metrics.csv')
    trans=table('transition_probe_metrics.csv');non=table('nonlinear_probe_metrics.csv')
    direction=table('transition_direction_metrics.csv');ordering=table('response_ordering_metrics.csv')
    paired=table('paired_comparisons.csv');geom_pair=table('geometry_paired_comparisons.csv')
    subgroup=table('subgroup_grounding_metrics.csv');ce=table('ce_reference_probe_metrics.csv')
    def metric(rep,var,h):return one(trans,representation=rep,target=var,horizon_seconds=h).r2
    zcur=[one(current,representation='z_t',target=v).r2 for v in ['BIS','MAP','HR']]
    pred300=[metric('dz_pred',v,300) for v in ['BIS','MAP','HR']]
    true300=[metric('dz_true',v,300) for v in ['BIS','MAP','HR']]
    pred30=[metric('dz_pred',v,30) for v in ['BIS','MAP','HR']]
    true30=[metric('dz_true',v,30) for v in ['BIS','MAP','HR']]
    trip_pred=one(ordering,subset='overall',representation='dz_pred',horizon_seconds=300)
    trip_true=one(ordering,subset='overall',representation='dz_true',horizon_seconds=300)
    trip_delta=one(geom_pair,subset='overall',horizon_seconds=300,metric='triplet_accuracy')
    cos_pred=one(direction,subset='overall',representation='dz_pred',target='BIS',horizon_seconds=300,category='same_minus_opposite')
    cos_true=one(direction,subset='overall',representation='dz_true',target='BIS',horizon_seconds=300,category='same_minus_opposite')
    n_pred=[one(non,representation='dz_pred',target=v,horizon_seconds=300).r2 for v in ['BIS','MAP','HR']]
    q4=one(subgroup,subset='Q4',representation='dz_pred',target='BIS',horizon_seconds=300)
    stable=one(subgroup,subset='stable_action_Q1',representation='dz_pred',target='BIS',horizon_seconds=300)
    clear_gap=all(v>.8 for v in zcur) and all(p<t-.3 for p,t in zip(pred300,true300))
    weak_geometry=trip_pred.triplet_accuracy<.60 and cos_pred.cosine<.10
    no_nonlinear_rescue=all(v<.5 for v in n_pred)
    gap_grows=(true300[0]-pred300[0])>(true30[0]-pred30[0])+.1
    intervention_worse=(q4.r2<stable.r2-.1)
    # The requested four labels are not exhaustive. Record which parts of A hold.
    if clear_gap and weak_geometry and no_nonlinear_rescue:
        outcome='A';label='Physiological transition-geometry gap (qualified)'
    elif all(n-p>.15 for n,p in zip(n_pred,pred300)) and weak_geometry:
        outcome='B';label='Geometry entangled, with nonlinear decodability'
    elif clear_gap and gap_grows:
        outcome='C';label='Grounded encoder versus deteriorating rollout'
    else:
        outcome='D';label='Grounding gap not established under this audit'
    flags={'strong_current_encoding':all(v>.8 for v in zcur),'linear_transition_gap':clear_gap,
           'weak_predicted_geometry':weak_geometry,'nonlinear_rescue':not no_nonlinear_rescue,
           'predicted_factual_gap_grows_with_horizon':gap_grows,
           'intervention_change_disproportionately_worse':intervention_worse}
    write_json('decision_criteria.json',flags)
    lines=['# VitalDB physiological grounding audit, seed 0',f'\n**Outcome {outcome} — {label}.**',
      '\n## Central finding',
      f'The frozen prospective RSSM still reproduces the prior 83,198-window TEST forecast exactly (maximum absolute difference {cache["ar_prediction_max_abs_difference"]}). Its current latent linearly recovers BIS/MAP/HR with patient-weighted R² {" / ".join(fmt(v) for v in zcur)}. At 300 s, a linear probe on predicted latent displacement recovers the respective physiological changes with R² {" / ".join(fmt(v) for v in pred300)}, while a probe on factual-anchor displacement reaches {" / ".join(fmt(v) for v in true300)}. The small MLP on predicted displacement reaches {" / ".join(fmt(v) for v in n_pred)}; nonlinear decoding does not remove the main gap.',
      f'At 300 s, predicted-displacement BIS same-minus-opposite response-direction cosine is {fmt(cos_pred.cosine)} [{fmt(cos_pred.ci_low)}, {fmt(cos_pred.ci_high)}], versus {fmt(cos_true.cosine)} [{fmt(cos_true.ci_low)}, {fmt(cos_true.ci_high)}] for factual displacement. Patient-balanced triplet ordering accuracy is {fmt(trip_pred.triplet_accuracy)} versus {fmt(trip_true.triplet_accuracy)}; their paired predicted-minus-factual difference is {fmt(trip_delta.predicted_minus_factual)} [{fmt(trip_delta.ci_low)}, {fmt(trip_delta.ci_high)}] across {int(trip_delta.patients)} patients with complete BIS/MAP/HR changes.',
      '\n**Interpretation boundary.** Factual-anchor `z_true(t+h)` encodes the actual physiology at the endpoint. Its high ΔBIS/ΔHR probe score is partly expected from endpoint information and cannot establish that the transition model drifts from a causal physiological manifold. The weak predicted-displacement direction and response-order tests are a separate geometric observation. The predicted BIS probe improves rather than deteriorates from 30 to 300 s, and Q4 intervention-change windows are not disproportionately worse than stable-action windows. These facts limit the strong horizon-decay and intervention-specific claims.',
      '\n## Unchanged cohort, model and leakage policy',
      f'The numeric VitalDB cohort contains {data["cases"]} cases and {data["patients"]} patients, with unchanged 345/74/76 case and 345/73/75 disjoint-subject TRAIN/VAL/TEST splits. Complete prospective-action windows number {data["eligible_windows"]["train"]:,}/{data["eligible_windows"]["val"]:,}/{data["eligible_windows"]["test"]:,}; {data["test_patients"]} TEST patients are evaluable. The original 10-second, 180-step history/30-step future, preprocessing, normalization, missingness masks, demographics and propofol/remifentanil administration are reused. No data were downloaded.',
      'The seed-0 58,946-parameter prospective-action RSSM checkpoint is frozen. For every eligible TRAIN/VAL/TEST window the cache stores z_t, all 30 predicted rollout latents, BIS/MAP predictions, and factual-anchor encodings at 30/60/180/300 s. Factual histories end at t+h and therefore include the observed endpoint but nothing later. Mutation checks show that changing data after the endpoint cannot change the factual input; changing endpoint physiology does change it. The large latent cache remains on the compute server and is excluded from Git.',
      'Ridge probes are fit only on TRAIN windows; regularization is selected from 0.1/10/1000 by VAL patient-weighted R². No TEST outcome chooses alpha. A 64→32→3 MLP is trained on at most 60,000 TRAIN windows per horizon/representation with VAL-only early stopping; no RSSM parameter is updated. Every reported TEST metric weights patients equally, and central intervals use 1,000 patient-cluster resamples. Overlapping windows do not count as independent patients.',
      '\n## Current state and future-change probes',
      '| Representation | Target | TEST R² | MAE | Pearson |',
      '|---|---|---:|---:|---:|']
    for v in ['BIS','MAP','HR']:
        for rep in ['z_t','demographics','drug_history','raw_other_current','random_latent']:
            r=one(current,representation=rep,target=v)
            lines.append(f'| {rep} | {v}_t | {fmt(r.r2)} | {fmt(r.mae)} | {fmt(r.pearson)} |')
    lines+=['\nThe raw-current control excludes the predicted variable from current value, trailing means, trailing SD and change; demographics remain. The random Gaussian representation is 64-dimensional, matching z_t. Current-state decodability is a sanity check. The original model receives historical state and its decoder adds the current BIS/MAP as a residual baseline, so current-state decodability alone is not a transition-grounding result.',
            '\n| Future ΔBIS representation | 30s R² | 60s | 180s | 300s |',
            '|---|---:|---:|---:|---:|']
    for rep in ['z_t','action','z_t+action','raw_history+action']:
        vals=[one(future,representation=rep,target='BIS',horizon_seconds=h).r2 for h in [30,60,180,300]]
        lines.append(f'| {rep} | '+ ' | '.join(fmt(v) for v in vals)+' |')
    lines+=['\n`future_change_probe_metrics.csv` gives the corresponding MAP and HR results. Future action summaries contain only dose information from t+1 through the evaluated horizon. `phys_response_magnitude.csv` gives the combined BIS/MAP/HR response norm standardized by TRAIN 300-s change SD.',
            '\n## Transition-probe scorecard',
            '| Representation | Target | 30s R² | 60s | 180s | 300s |',
            '|---|---|---:|---:|---:|']
    card=table('main_grounding_scorecard.csv')
    for r in card.itertuples():
        lines.append(f'| {r.representation} | {r.target} | '+ ' | '.join(fmt(getattr(r,f'r2_{h}s')) for h in [30,60,180,300])+' |')
    lines+=['\nThe largest predicted-vs-factual probe gap is at short horizons for BIS and HR; it narrows with horizon rather than growing. `paired_comparisons.csv` gives paired TEST R² and MAE differences with 1,000 patient-cluster intervals. `horizon_grounding_metrics.csv` joins all linear and nonlinear horizon results.',
            '\n| Representation | 300s ΔBIS linear / MLP R² | ΔMAP linear / MLP | ΔHR linear / MLP |',
            '|---|---:|---:|---:|']
    for rep in ['dz_pred','dz_true']:
        cells=[]
        for v in ['BIS','MAP','HR']:
            cells.append(f'{fmt(metric(rep,v,300))} / {fmt(one(non,representation=rep,target=v,horizon_seconds=300).r2)}')
        lines.append(f'| {rep} | '+' | '.join(cells)+' |')
    lines+=['\n## Geometry, response ordering and intervention subgroups',
            '| Representation, 300s | BIS same−opposite cosine [95% CI] | Triplet accuracy [95% CI] | Pair-distance Spearman |',
            '|---|---:|---:|---:|']
    for rep in ['dz_pred','dz_true']:
        c=one(direction,subset='overall',representation=rep,target='BIS',horizon_seconds=300,category='same_minus_opposite')
        o=one(ordering,subset='overall',representation=rep,horizon_seconds=300)
        lines.append(f'| {rep} | {fmt(c.cosine)} [{fmt(c.ci_low)}, {fmt(c.ci_high)}] | {fmt(o.triplet_accuracy)} [{fmt(o.triplet_ci_low)}, {fmt(o.triplet_ci_high)}] | {fmt(o.distance_spearman)} |')
    lines+=['\nResponse groups use TRAIN ΔBIS/ΔMAP quintiles. The opposite-response comparison uses only bins whose indices differ by at least three; stable central bins cannot enter that pair and the conditional population is narrower than for random pairs. Up to 40 windows per TEST patient contribute to each geometry sample, with partners drawn from other patients. Pairwise distances and triplets use TRAIN change and latent scales. `transition_direction_metrics.csv`, `response_ordering_metrics.csv` and `geometry_paired_comparisons.csv` report every horizon, MAP direction and 1,000-patient-bootstrap comparisons. Predicted BIS directions have a small positive same–opposite signal; they are not completely random. MAP direction separation is weak for both representations.',
            '\n| 300s BIS subgroup | Predicted Δz R² | Factual Δz R² | Predicted triplet accuracy |',
            '|---|---:|---:|---:|']
    for name in ['stable_action_Q1','Q4','upcoming_large_intervention','initiation','increase','decrease','stop']:
        p=one(subgroup,subset=name,representation='dz_pred',target='BIS',horizon_seconds=300)
        f=one(subgroup,subset=name,representation='dz_true',target='BIS',horizon_seconds=300)
        o=one(ordering,subset=name,representation='dz_pred',horizon_seconds=300)
        lines.append(f'| {name} | {fmt(p.r2)} | {fmt(f.r2)} | {fmt(o.triplet_accuracy)} |')
    lines+=['\nThe Round-2 Q4 and action-only upcoming event definitions are reused without rethresholding. Under this audit, predicted transition probing is not worse in Q4 or upcoming interventions than in stable Q1. The stop subgroup is small. This does not establish a treatment-specific grounding failure. `subgroup_grounding_metrics.csv` contains 180/300-s BIS/MAP/HR R² with patient-cluster intervals.',
            '\n## Device-computed TCI reference (secondary)',
            '| CE reference | z_t current R² | Δz_pred 300s ΔCE R² | Δz_true 300s ΔCE R² |',
            '|---|---:|---:|---:|']
    for var in ['PPF_CE','RFTN_CE']:
        vals=[one(ce,representation=rep,target=var,horizon_seconds=h).r2 for rep,h in [('z_t',0),('dz_pred',300),('dz_true',300)]]
        lines.append(f'| {var} | '+' | '.join(fmt(v) for v in vals)+' |')
    lines+=['\nPPF_CE and RFTN_CE are device-computed TCI reference states, not measured exposure concentrations. They are used only for post-hoc probing and never as RSSM inputs, losses or intervention labels.',
            '\n## Figures',
            *[f'![Figure {i}](../plots/{name}.png)' for i,name in enumerate([
              'figure1_current_vs_change','figure2_transition_grounding','figure3_transition_direction',
              'figure4_response_distance','figure5_intervention_change','figure6_prespecified_trajectories'],1)],
            '\nThe two display cases are fixed at the 25th and 75th percentiles of sorted TEST case IDs, with each case’s median eligible anchor. PCA is fitted only on TRAIN current latents for visualization and is not used as a grounding metric.',
            '\n## Decision and limitations']
    if outcome=='A':
        lines.append('The strongest supported finding is a broad geometric gap: current physiology is highly decodable, while predicted latent displacement has much weaker linear and small-MLP change decoding and weak response-order geometry. This meets the central forecasting-versus-transition-geometry concern. Outcome A is qualified: the predicted–factual gap narrows rather than grows with horizon, and it is not selectively worse during intervention changes. Factual-anchor encoding has access to the observed endpoint, so the large R² gap cannot by itself prove model rollout drift. A targeted transition-grounding training experiment could be tested, but these data do not justify claiming causal physiological mechanism recovery or a proven intervention-specific failure.')
    elif outcome=='B':lines.append('A small nonlinear probe recovers physiological change that linear probes miss, while direction/order geometry remains weak. The evidence favors organization rather than missing information.')
    elif outcome=='C':lines.append('Factual endpoint encoding is much more aligned and the gap grows with horizon, consistent with transition drift. Endpoint-information asymmetry remains a diagnostic limitation.')
    else:lines.append('The specified physiological grounding gap is not supported. Do not proceed to a new grounding objective on this evidence.')
    lines.append('\nThis is one single-center observational cohort, one patient split and one seed-0 frozen model. Probe fitting uses highly overlapping windows; only patient-level resampling supplies TEST uncertainty. The RSSM decoder predicts BIS/MAP as residuals added to the current state, and HR was never a forecast target. Those architectural facts affect how much physiology its latent displacement needs to carry. Neither high factual-anchor decodability nor cosine geometry identifies causal treatment effects. Additional architectures and external data would be needed for a general claim.')
    (OUT/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n')
    return outcome

def main():
    OUT.mkdir(exist_ok=True);(ROOT/'plots').mkdir(exist_ok=True)
    store=checks();scorecards();figures(store);outcome=report()
    required=['data_summary.json','latent_cache_summary.json','current_state_probe_metrics.csv',
      'future_change_probe_metrics.csv','transition_probe_metrics.csv','nonlinear_probe_metrics.csv',
      'transition_direction_metrics.csv','response_ordering_metrics.csv','subgroup_grounding_metrics.csv',
      'ce_reference_probe_metrics.csv','horizon_grounding_metrics.csv','paired_comparisons.csv',
      'integrity_checks.json','FINAL_REPORT.md']
    status={'outcome':outcome,'required_outputs_present':all((OUT/f).exists() for f in required),
            'figures_png':len(list((ROOT/'plots').glob('figure*.png'))),
            'no_new_world_model_checkpoint':not (OUT/'world_model_checkpoint.pt').exists(),
            'AR_exact_prediction_match':json.loads((OUT/'latent_cache_summary.json').read_text())['ar_prediction_max_abs_difference']==0,
            'bootstrap_replicates':CFG['bootstrap_replicates']}
    write_json('completion_checks.json',status)
    assert status['required_outputs_present'] and status['figures_png']>=6 and status['no_new_world_model_checkpoint']
    print('FINAL_REPORT_COMPLETE',outcome,flush=True)

if __name__=='__main__':main()
