"""Reconstruct tables/figures and evidence-led final report from saved ledgers."""
import argparse,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core import *

OUT=ROOT/'outputs';PLOTS=ROOT/'plots';PLOTS.mkdir(exist_ok=True)
COLORS={'Historical RSSM':'#666666','Prospective true':'#0072B2',
        'Prospective hold':'#E69F00','Prospective wrong':'#D55E00'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,'figure.dpi':130,
                     'savefig.dpi':180,'pdf.fonttype':42})

def save(fig,name):
    fig.tight_layout();fig.savefig(PLOTS/(name+'.png'),bbox_inches='tight')
    fig.savefig(PLOTS/(name+'.pdf'),bbox_inches='tight');plt.close(fig)

def label(row):
    return 'Historical RSSM' if row.model=='Historical RSSM' else 'Prospective '+row.condition

def fmt(x): return 'NA' if pd.isna(x) else f'{x:.3f}'

def md(df):
    d=df.copy()
    for c in d:
        if pd.api.types.is_float_dtype(d[c]):d[c]=d[c].map(fmt)
    return '| '+' | '.join(d.columns)+' |\n| '+' | '.join(['---']*len(d.columns))+' |\n'+\
           '\n'.join('| '+' | '.join(map(str,row))+' |' for row in d.itertuples(index=False,name=None))

def main_table(forecast,paired):
    rows=[];f=forecast[(forecast.target=='BIS')&(forecast.horizon_seconds==0)]
    for model,condition in [('Historical RSSM','hold'),('Prospective RSSM','true'),
                            ('Prospective RSSM','hold'),('Prospective RSSM','wrong')]:
        d=f[(f.model==model)&(f.condition==condition)]
        row={'Model / Action Condition':'Historical RSSM / hold' if model=='Historical RSSM' else 'Prospective RSSM / '+condition}
        for sub,name in [('overall','Overall 5m MAE'),('large_transition','Large-change 5m MAE'),
                         ('initiation','Initiation'),('increase','Increase'),('decrease','Decrease'),('stop','Stop')]:
            v=d[d.subset==sub];row[name]=float(v.mae.iloc[0]) if len(v) else np.nan
        if model=='Prospective RSSM' and condition=='true':
            for metric in ['FAV','MFAV']:
                v=paired[(paired.comparison==metric)&(paired.subset=='large_transition')&(paired.horizon_seconds==0)]
                row[metric]=float(v.difference_mae.iloc[0]) if len(v) else np.nan
        else:row.update({'FAV':np.nan,'MFAV':np.nan})
        rows.append(row)
    table=pd.DataFrame(rows);table.to_csv(OUT/'main_table.csv',index=False)
    return table

def figure_horizons(forecast,subset,name):
    d=forecast[(forecast.target=='BIS')&(forecast.subset==subset)&(forecast.horizon_seconds>0)]
    fig,ax=plt.subplots(figsize=(7,4.2))
    for key,color in COLORS.items():
        v=d[d.apply(label,axis=1)==key].sort_values('horizon_seconds')
        if len(v):
            x=v.horizon_seconds.to_numpy()/60;y=v.mae.to_numpy()
            ax.plot(x,y,'o-',label=key,color=color,lw=2)
            ax.fill_between(x,v.ci_low,v.ci_high,color=color,alpha=.11)
    ax.set(xlabel='Forecast horizon (minutes)',ylabel='BIS MAE',title=subset.replace('_',' ').title())
    ax.legend(frameon=False,fontsize=8);save(fig,name)

def figure_divergence(paired,protocol):
    d=paired[(paired.comparison=='FAV')&(paired.horizon_seconds==0)&(paired.subset.isin(['Q1','Q2','Q3','Q4']))]
    fig,ax=plt.subplots(figsize=(6,4));x=np.arange(4)
    v=d.set_index('subset').reindex([f'Q{i}' for i in range(1,5)])
    y=v.difference_mae.to_numpy(float);lo=v.ci_low.to_numpy(float);hi=v.ci_high.to_numpy(float)
    ax.bar(x,np.nan_to_num(y,nan=0),color='#0072B2',alpha=.8)
    for i in range(4):
        if np.isfinite(y[i]):ax.errorbar(i,y[i],yerr=[[max(0,y[i]-lo[i])],[max(0,hi[i]-y[i])]],fmt='none',color='black',capsize=3)
    ax.axhline(0,color='black',lw=.8);ax.set_xticks(x,[f'Q{i}' for i in range(1,5)])
    ax.set(ylabel='FAV = MAE(hold) - MAE(true)',xlabel='Train-derived future-action divergence bin',
           title='Future-action value by schedule divergence')
    for i,row in enumerate(v.itertuples()):
        n=0 if pd.isna(row.n_windows) else int(row.n_windows)
        ax.text(i,max(0,np.nan_to_num(y[i]))+.02,f'n={n:,}',ha='center',fontsize=8)
    save(fig,'figure3_future_action_divergence')

def figure_kinds(paired):
    names=['initiation','increase','decrease','stop']
    d=paired[(paired.comparison=='FAV')&(paired.horizon_seconds==0)].set_index('subset').reindex(['upcoming_'+x for x in names])
    fig,ax=plt.subplots(figsize=(7,4));x=np.arange(4)
    y=d.difference_mae.to_numpy(float);lo=d.ci_low.to_numpy(float);hi=d.ci_high.to_numpy(float)
    ax.bar(x,y,color='#009E73',alpha=.85)
    for i in range(4):
        if np.isfinite(y[i]):ax.errorbar(i,y[i],yerr=[[max(0,y[i]-lo[i])],[max(0,hi[i]-y[i])]],fmt='none',color='black',capsize=3)
    ax.axhline(0,color='black',lw=.8);ax.set_xticks(x,[s.title() for s in names])
    ax.set(ylabel='FAV = MAE(hold) - MAE(true)',title='Upcoming large-change confirmations (action-only labels)')
    save(fig,'figure4_event_type_FAV')

def figure_patients(store,ref):
    cap=np.array(json.loads((OUT/'pump_rate_tail_protocol.json').read_text())['0.99']['maximum_ml_per_10s'])
    current=np.array([store.cases[int(cid)]['action'][int(t)] for cid,t in zip(ref['case'],ref['t'])])
    clean=(ref['future_action']<=cap).all((1,2))&(current<=cap).all(1)
    eligible=(ref['donor']>=0)&(ref['quartile']==4)&clean
    cases=sorted(np.unique(ref['case'][eligible]).tolist())
    assert len(cases)>=3
    chosen=[cases[int((len(cases)-1)*q)] for q in [.15,.5,.85]]
    predictions={k:np.load(OUT/f'prediction_{k}.npz')['prediction'] for k in ['true','hold','wrong']}
    fig,axes=plt.subplots(3,2,figsize=(12,9),gridspec_kw={'width_ratios':[2,1]})
    for row,cid in enumerate(chosen):
        candidates=np.flatnonzero((ref['case']==cid)&eligible)
        if not len(candidates):continue
        # Median schedule divergence among clean Q4 windows, independent of outcomes/errors.
        order=candidates[np.argsort(ref['divergence'][candidates])]
        i=int(order[len(order)//2]);t=int(ref['t'][i]);c=store.cases[cid]
        time=np.arange(31)*10/60
        ax=axes[row,0];ax.plot(time,c['state'][t:t+31,0],label='Observed BIS',color='black',lw=2)
        for k,color in [('true','#0072B2'),('hold','#E69F00'),('wrong','#D55E00')]:
            ax.plot(time[1:],predictions[k][i,:,0],label=k,color=color)
        ax.set(xlabel='Minutes after anchor',ylabel='BIS',title=f'Case {cid}; action-selected anchor t={t}')
        if row==0:ax.legend(frameon=False,ncol=4,fontsize=8)
        bx=axes[row,1]
        for j,drug,color in [(0,'Propofol','#0072B2'),(1,'Remifentanil','#D55E00')]:
            bx.step(time[1:],ref['future_action'][i,:,j]*360,where='mid',color=color,label=drug)
            bx.axhline(c['action'][t,j]*360,color=color,ls=':',alpha=.6)
        bx.set(xlabel='Minutes after anchor',ylabel='Administration (mL/h)',title='True future schedule; dotted = hold')
        if row==0:bx.legend(frameon=False,fontsize=8)
    save(fig,'figure5_representative_rollouts')

def figure_exposure_probe(probe):
    d=probe[probe.horizon_seconds==0]
    fig,axes=plt.subplots(1,2,figsize=(8,3.8),sharey=True)
    for ax,drug in zip(axes,['PPF_CE','RFTN_CE']):
        q=d[d.drug==drug].set_index('condition').reindex(['true','hold','wrong'])
        ax.bar(q.index,q.r2.to_numpy(),color=['#0072B2','#E69F00','#D55E00'])
        ax.set(title=drug,ylabel='Patient-weighted R²',xlabel='Future action schedule')
    save(fig,'figure6_future_exposure_probe')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--outcome',choices=['A','B','C','D'])
    parser.add_argument('--interpretation-file');args=parser.parse_args()
    f=pd.read_csv(OUT/'forecast_metrics.csv');p=pd.read_csv(OUT/'paired_comparisons.csv')
    oracle=pd.read_csv(OUT/'future_action_informativeness.csv')
    trend=pd.read_csv(OUT/'divergence_trend.csv')
    tail=pd.read_csv(OUT/'pump_rate_tail_sensitivity.csv')
    tail_protocol=json.loads((OUT/'pump_rate_tail_protocol.json').read_text())
    amendments=(OUT/'protocol_amendments.md').read_text(encoding='utf8')
    protocol=json.loads((OUT/'divergence_protocol.json').read_text())
    summary=json.loads((OUT/'data_summary.json').read_text())
    checks=json.loads((OUT/'integrity_checks.json').read_text())
    models=pd.read_csv(OUT/'model_summary.csv');control_models=pd.read_csv(OUT/'informativeness_model_summary.csv')
    table=main_table(f,p)
    figure_horizons(f,'overall','figure1_overall_horizons')
    figure_horizons(f,'large_transition','figure2_large_change_horizons')
    figure_divergence(p,protocol);figure_kinds(p)
    store=Store();ref=np.load(OUT/'test_reference.npz');figure_patients(store,ref)
    ce_path=OUT/'rollout_ce_probe_metrics.csv'
    if ce_path.exists():
        ce_probe=pd.read_csv(ce_path);figure_exposure_probe(ce_probe)
    if args.outcome:
        if not args.interpretation_file:raise ValueError('Interpretation file required')
        interpretation=open(args.interpretation_file,encoding='utf8').read()
        title={'A':'Prospective controllability gap strongly supported',
               'B':'Partial prospective controllability gap','C':'Gap not supported',
               'D':'Inconclusive / dataset does not identify the question'}[args.outcome]
        main=f[(f.target=='BIS')&(f.horizon_seconds==0)&(f.subset.isin(['overall','large_transition','high_future_divergence']))]
        central=p[(p.horizon_seconds==0)&(p.subset.isin(['overall','large_transition','upcoming_large_transition','high_future_divergence']))]
        quantiles=p[(p.horizon_seconds==0)&(p.comparison=='FAV')&(p.subset.isin(['Q1','Q2','Q3','Q4']))]
        events=p[(p.horizon_seconds==0)&(p.comparison=='FAV')&(p.subset.isin(['initiation','increase','decrease','stop']))]
        upcoming=p[(p.horizon_seconds==0)&(p.comparison=='FAV')&(p.subset.isin(['upcoming_initiation','upcoming_increase','upcoming_decrease','upcoming_stop']))]
        informative=oracle[oracle.horizon_seconds==0]
        ce_section=''
        if ce_path.exists():
            ce_section='''## Frozen future-latent TCI-reference probe

The forecasting model is frozen. Ridge probes are fit on sampled TRAIN future latent trajectories to device-computed TCI CE at each future step and evaluated on TEST patients. They receive no gradients through the forecaster. This is a secondary exposure consistency audit, not measured concentration or causal fidelity.\n\n'''+md(ce_probe[ce_probe.horizon_seconds==0])+'''\n\n![Future exposure probe](../plots/figure6_future_exposure_probe.png)\n'''
        report=f'''# VitalDB prospective-action diagnostic, seed 0

**Outcome {args.outcome} — {title}.**

{interpretation}

## Question and design

Can a history encoder with an action-conditioned latent transition use a prospective propofol/remifentanil administration schedule to predict BIS? This is conditional prediction on observed treatment choices, not identification of causal treatment effects. The first-round CE representation result motivated this question but is not reused as supervision. Both RSSMs use the exact same 58,946-parameter architecture, initialization seed, patient splits, training windows and per-epoch training indices. The only distinction is whether every rollout GRUCell receives the last historical action or the observed future action at that step. Both were trained from scratch under their respective schedules. Two matched-capacity direct supervised controls assess whether future actions are predictively informative under another model class.

## Data and integrity

Source: first experiment's VitalDB numeric case arrays. Exactly {summary['included_cases']} source cases and {summary['included_patients']} source patients; same train/validation/test IDs. Original windows and complete-future-action windows: `{json.dumps(summary['windows'])}`. One of the 75 held-out patients has no window with a complete future-action sequence, so primary analysis contains {summary['test_patients_with_complete_future_action']} test patients. Future medication is directly conditioned from t+1 through t+30; future BIS/MAP never enters features. Windows with an incomplete future medication sequence are omitted from **both** models and all comparisons. Both actions use the original RATE-preferred or cumulative-VOL fallback mL/10s trajectories and original train-only normalization. Missing past values have explicit masks. TCI CE is reference-only.

Automated checks: `{json.dumps(checks)}`. Matched-wrong future schedules are selected from other TEST patients by current BIS/HR/MAP within 0.5 training SD and future dose-sequence RMS distance >=0.25 training action SD. Matching never uses future physiology or CE. Matched coverage: {summary['test_matched_windows']} / {summary['windows']['test']['complete_future_action']} test windows.

The inherited Round-1 large-change labels are **confirmed historical events at or before t** (the 2-minute post-confirmation window). They are retained as the specified ~18k subset, but are not automatically upcoming intervention changes. A separate action-only upcoming label requires a large-change confirmation from t+60 to t+300 seconds, reducing the chance that the change began before t. The high-future-divergence and train-derived Q1–Q4 analyses directly target prospective variation. Quantile protocol: `{json.dumps(protocol)}`. Empty bins, if any, reflect tied train quantiles and are not merged after test inspection.

## Models and training

10-second samples; 180 historical steps; 30 future steps; BIS primary and arterial MAP secondary. Standardized BIS MSE + 0.25 MAP MSE, AdamW, fixed seed 0, 80,000 distinct shared train windows per epoch, up to 20 epochs and validation-only early stopping. No CE loss, hyperparameter search or architecture modification of the central RSSM. Model summary:

{md(models)}

Direct supervised informativeness controls:

{md(control_models)}

## Main table

The MAE columns below are patient-weighted averages over the entire 5-minute BIS trajectory. FAV and MFAV are paired differences on the large-change subset; matched-wrong comparisons use matched windows only.

{md(table)}

## Horizons and paired comparisons

Point errors are at t+30, 60, 180 and 300 seconds. `horizon_seconds=0` denotes full five-minute trajectory. All intervals use 1,000 patient-cluster bootstrap replicates; patients contribute equal weight, preserving within-patient window dependence.

{md(main[['model','condition','subset','mae','ci_low','ci_high','n_windows','n_patients']])}

{md(central[['comparison','subset','difference_mae','ci_low','ci_high','n_windows','n_patients']])}

![Overall horizons](../plots/figure1_overall_horizons.png)
![Large-change horizons](../plots/figure2_large_change_horizons.png)

## Future-action divergence and event types

{md(quantiles[['subset','difference_mae','ci_low','ci_high','n_windows','n_patients']])}

Patient-level trend tests (1,000 cluster bootstrap replicates):

{md(trend)}

{md(events[['subset','difference_mae','ci_low','ci_high','n_windows','n_patients']])}

Upcoming confirmed event types:

{md(upcoming[['subset','difference_mae','ci_low','ci_high','n_windows','n_patients']])}

![Divergence](../plots/figure3_future_action_divergence.png)
![Event types](../plots/figure4_event_type_FAV.png)

### High-rate episode sensitivity

Some Q4 schedules contain brief high pump rates. As an exploratory check, exclude any window where the current or future dose exceeds the TRAIN positive-rate 99th or 99.5th percentile for either drug. This removes potentially informative bolus-like episodes as well as possible rate artifacts, so a reduced FAV does not by itself establish model failure. Cutoffs and retained window counts: `{json.dumps(tail_protocol)}`. The direct-control value is recomputed on exactly the same restricted windows.

{md(tail)}

## Future-action informativeness control

The two generic direct supervised models share architecture, parameters, initialization, optimizer, training windows and validation. One gets historical features only; the other additionally receives the observed future action sequence. Their patient-paired difference measures predictive information available to this supervised class, not a causal treatment effect.

{md(informative)}

## Representative patient schedules

Cases are fixed by sorted percentiles among test cases with a matched Q4 window below the TRAIN positive-rate 99th-percentile dose cap. Within each case, the plotted anchor has median divergence among such windows. Selection uses only dose schedules and matching, never physiological outcomes or model errors. This avoids choosing transient extreme-rate episodes just for visual drama.

![Patient rollouts](../plots/figure5_representative_rollouts.png)

{ce_section}

## Protocol and implementation amendments

{amendments}

## Interpretation and limits

The 5-minute horizon may be short relative to pharmacodynamic effects. Device-controlled TCI administration and unmeasured concurrent actions confound observational schedule-response interpretation. The data are one center and one seed. The labels of the Round-1 large-change subset refer to recently confirmed changes, so direct claims about upcoming changes must rely on divergence and future-event analyses. Even the upcoming confirmations use a trailing median and do not precisely timestamp action onset. TCI CE is computed by the device and is not measured exposure. A predictive gain from known future treatment does not prove physiological causal controllability; likewise, no gain in a dataset without direct future-action information does not establish model failure.

**Final evidence classification: Outcome {args.outcome} — {title}.**
'''
        (OUT/'FINAL_REPORT.md').write_text(report,encoding='utf8')
    print('REPORT_READY',flush=True)

if __name__=='__main__':main()
