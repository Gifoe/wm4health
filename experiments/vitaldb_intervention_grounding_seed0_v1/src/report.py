"""Tables and figures from recorded results; conclusion must be supplied after review."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core import ROOT,Store,filename,MODEL_NAMES

OUT=ROOT/'outputs'; PLOTS=ROOT/'plots'
COLORS={'Persistence':'#999999','State-only':'#525252','Action Transformer':'#0072B2','RSSM':'#D55E00',
        'Current physiology':'#009E73','Raw drug history':'#CC79A7'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                    'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,'figure.dpi':130,
                    'savefig.dpi':180,'pdf.fonttype':42})

def save(fig,name):
    fig.tight_layout(); fig.savefig(PLOTS/(name+'.png'),bbox_inches='tight')
    fig.savefig(PLOTS/(name+'.pdf'),bbox_inches='tight'); plt.close(fig)

def markdown(df,precision=3):
    f=df.copy()
    for c in f.columns:
        if pd.api.types.is_float_dtype(f[c]): f[c]=f[c].map(lambda v:f'{v:.{precision}f}' if pd.notna(v) else 'NA')
    header='| '+' | '.join(map(str,f.columns))+' |\n'
    header+='| '+' | '.join(['---']*len(f.columns))+' |\n'
    return header+'\n'.join('| '+' | '.join(str(v) for v in row)+' |' for row in f.itertuples(index=False,name=None))

def tables(forecast,probe,timing,ablation):
    results={}
    for sub in ['overall','transition','large_transition']:
        rows=[]
        for name in ['Persistence']+MODEL_NAMES:
            row={'Model':name}
            ff=forecast[(forecast.model==name)&(forecast.subset==sub)&(forecast.target=='BIS')]
            for h,label in [(30,'30s'),(60,'1m'),(180,'3m'),(300,'5m')]:
                ss=ff[ff.horizon_seconds==h]; row['BIS MAE '+label]=ss.mae.iloc[0] if len(ss) else np.nan
            for drug in ['PPF_CE','RFTN_CE']:
                p=probe[(probe.representation==name)&(probe.subset==sub)&(probe.drug==drug)]
                row[drug+' R2']=p.r2.iloc[0] if len(p) else np.nan
            for intervention,label in [('zero_action','ARD'),('matched_shuffle','MWAD')]:
                a=ablation[(ablation.model==name)&(ablation.subset==sub)&(ablation.intervention==intervention)&(ablation.horizon_seconds==0)]
                row[label]=a.degradation.iloc[0] if len(a) else np.nan
            t=timing[(timing.model==name)&(timing.subset==sub)&(timing.horizon_seconds==0)]
            row['Timing max |TD| %']=100*t.normalized_degradation.abs().max() if len(t) else np.nan
            rows.append(row)
        results[sub]=pd.DataFrame(rows)
        table_name={'overall':'main_diagnostic_table.csv','transition':'transition_diagnostic_table.csv','large_transition':'large_transition_diagnostic_table.csv'}[sub]
        results[sub].to_csv(OUT/table_name,index=False)
    return results

def render(forecast,probe,timing,ablation):
    PLOTS.mkdir(exist_ok=True)
    fig,axs=plt.subplots(1,2,figsize=(10,3.8))
    for ax,target in zip(axs,['BIS','MAP']):
        for name in ['Persistence']+MODEL_NAMES:
            d=forecast[(forecast.subset=='overall')&(forecast.model==name)&(forecast.target==target)&(forecast.horizon_seconds>0)].sort_values('horizon_seconds')
            ax.plot(d.horizon_seconds/60,d.mae,'o-',label=name,color=COLORS[name],lw=1.8,ms=4)
            ax.fill_between(d.horizon_seconds/60,d.mae_ci_low,d.mae_ci_high,color=COLORS[name],alpha=.12)
        ax.set(xlabel='Forecast horizon (minutes)',ylabel=f'{target} MAE'+(' (mmHg)' if target=='MAP' else ' (points)'),title=target+' forecast')
    axs[0].legend(frameon=False,fontsize=8)
    fig.suptitle('Held-out patients · equal patient weighting · 95% patient bootstrap CI',fontsize=11)
    save(fig,'figure1_forecasting')
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    reps=['Current physiology','Raw drug history','Action Transformer','RSSM']
    for ax,drug in zip(axs,['PPF_CE','RFTN_CE']):
        for i,name in enumerate(reps):
            p=probe[(probe.representation==name)&(probe.drug==drug)&(probe.subset=='overall')].iloc[0]
            ax.bar(i,p.r2,color=COLORS[name],width=.7)
            ax.errorbar(i,p.r2,yerr=[[max(0,p.r2-p.r2_ci_low)],[max(0,p.r2_ci_high-p.r2)]],fmt='none',ecolor='#222222',capsize=3)
        ax.set_xticks(range(4),['Current\nphysiology','Raw drug\nhistory','Transformer\nlatent','RSSM\nlatent'])
        ax.set(ylabel='Linear-probe R²',title=drug.replace('_',' ')); ax.axhline(0,color='#333333',lw=.7)
    fig.suptitle('Device-computed TCI reference states · probes fit on train patients only',fontsize=11)
    save(fig,'figure2_ce_probes')
    fig,axs=plt.subplots(2,2,figsize=(10,7))
    for col,sub in enumerate(['overall','transition']):
        ax=axs[0,col]; relative=axs[1,col]
        for name in MODEL_NAMES[1:]:
            d=timing[(timing.model==name)&(timing.subset==sub)&(timing.horizon_seconds==0)].sort_values('shift_seconds')
            ax.plot(d.shift_seconds,d.mae,'o-',color=COLORS[name],label=name)
            baseline=float(d[d.shift_seconds==0].mae.iloc[0])
            relative.plot(d.shift_seconds,100*d.normalized_degradation,'o-',color=COLORS[name],label=name)
            relative.fill_between(d.shift_seconds,100*d.delta_ci_low/baseline,100*d.delta_ci_high/baseline,color=COLORS[name],alpha=.15)
        ax.axvline(0,color='#777777',ls=':',lw=.8)
        ax.set(xlabel='Drug-history shift Δ (seconds; + means delayed)',ylabel='Full-trajectory BIS MAE',title=sub.title())
        ax.set_ylim(bottom=0)
        ax.set_xticks([-120,-60,0,60,120])
        relative.axhline(0,color='#555555',lw=.8)
        relative.axhspan(-2,2,color='#888888',alpha=.07,label='±2% descriptive band')
        relative.set(xlabel='Drug-history shift Δ (seconds)',ylabel='Paired MAE change / true MAE (%)')
        relative.set_xticks([-120,-60,0,60,120])
    axs[0,0].legend(frameon=False)
    axs[1,0].legend(frameon=False,fontsize=8)
    save(fig,'figure3_timing_sensitivity')
    fig,axs=plt.subplots(1,3,figsize=(14,4))
    for name in ['Persistence']+MODEL_NAMES:
        d=forecast[(forecast.model==name)&(forecast.subset=='transition')&(forecast.target=='BIS')&(forecast.horizon_seconds>0)].sort_values('horizon_seconds')
        axs[0].plot(d.horizon_seconds/60,d.mae,'o-',label=name,color=COLORS[name])
    axs[0].set(xlabel='Horizon (minutes)',ylabel='BIS MAE',title='Dose-change windows'); axs[0].legend(fontsize=7,frameon=False)
    for k,drug in enumerate(['PPF_CE','RFTN_CE']):
        x=np.arange(4)+k*.36
        d=probe[(probe.drug==drug)&(probe.subset=='transition')].set_index('representation').reindex(reps)
        axs[1].bar(x,d.r2,width=.33,label=drug)
    axs[1].set_xticks(np.arange(4)+.18,['Physiology','Drug history','Transformer','RSSM'],rotation=20)
    axs[1].set(ylabel='CE probe R²',title='Transition reference recovery'); axs[1].legend(fontsize=7,frameon=False)
    for i,name in enumerate(MODEL_NAMES[1:]):
        for k,intervention in enumerate(['zero_action','matched_shuffle']):
            d=ablation[(ablation.model==name)&(ablation.subset=='transition')&(ablation.intervention==intervention)&(ablation.horizon_seconds==0)].iloc[0]
            x=i*3+k
            axs[2].bar(x,d.degradation,color=COLORS[name],alpha=1 if k==0 else .6)
            axs[2].errorbar(x,d.degradation,yerr=[[max(0,d.degradation-d.delta_ci_low)],[max(0,d.delta_ci_high-d.degradation)]],fmt='none',ecolor='black',capsize=3)
    axs[2].set_xticks([0,1,3,4],['T: zero','T: shuffle','R: zero','R: shuffle'],rotation=25)
    axs[2].set(ylabel='BIS MAE increase (paired)',title='Action controls at transitions')
    save(fig,'figure4_transition_audit')
    ref=np.load(OUT/'test_reference.npz'); store=Store()
    ids=sorted(store.splits['test']); chosen=[ids[int(q*(len(ids)-1))] for q in [.15,.5,.85]]
    preds={name:np.load(OUT/f'predictions_{filename(name)}.npz')['prediction'] for name in MODEL_NAMES}
    ce_preds={name:np.load(OUT/f'ce_prediction_{filename(name)}.npy') for name in MODEL_NAMES[1:]}
    fig,axs=plt.subplots(4,3,figsize=(15,10),squeeze=False)
    for col,cid in enumerate(chosen):
        c=store.cases[cid]; time=c['time']/60; rows=np.flatnonzero(ref['case']==cid); t=ref['t'][rows]
        axs[0,col].plot(time,c['state'][:,0],color='#444444',lw=1,label='Observed BIS')
        sample=rows[::30]
        for name in MODEL_NAMES:
            axs[0,col].plot((ref['t'][sample]+30)/6,preds[name][sample,29,0],'.-',color=COLORS[name],ms=3,lw=.8,label=name+' (+5 min)')
        axs[0,col].set_title(f'Case {cid}'); axs[0,col].set_ylabel('BIS')
        for j,label in enumerate(['Propofol','Remifentanil']):
            axs[1,col].plot(time,c['action'][:,j]*360,lw=.8,label=label)
        axs[1,col].set_ylabel('Pump rate (mL/h)')
        for j,label in enumerate(['PPF CE (µg/mL)','RFTN CE (ng/mL)']):
            ax=axs[j+2,col]; ax.plot(time,c['ce'][:,j],color='#444444',lw=1,label='TCI reference')
            for name in MODEL_NAMES[1:]: ax.plot(t[::6]/6,ce_preds[name][rows[::6],j],lw=.9,color=COLORS[name],label=name+' probe')
            ax.set_ylabel(label); ax.set_xlabel('Case numeric timeline (minutes)')
    for row in range(4): axs[row,0].legend(fontsize=6,frameon=False,loc='best')
    fig.suptitle('Patients selected by case-ID quantiles before viewing model errors',fontsize=13)
    save(fig,'figure5_patient_trajectories')
    fig,axs=plt.subplots(2,2,figsize=(9,8))
    for i,name in enumerate(MODEL_NAMES[1:]):
        for j,drug in enumerate(['Propofol CE','Remifentanil CE']):
            ax=axs[i,j]; yy=ref['ce'][:,j]; pp=ce_preds[name][:,j]; ok=np.isfinite(yy)&np.isfinite(pp)
            if ok.sum()>12000:
                inds=np.random.default_rng(0).choice(np.flatnonzero(ok),12000,replace=False)
            else: inds=np.flatnonzero(ok)
            ax.hexbin(yy[inds],pp[inds],gridsize=45,mincnt=1,cmap='Blues',bins='log')
            lo=min(np.nanmin(yy[inds]),np.nanmin(pp[inds])); hi=max(np.nanmax(yy[inds]),np.nanmax(pp[inds]))
            ax.plot([lo,hi],[lo,hi],ls='--',lw=.8,color='#D55E00')
            ax.set(xlabel='TCI reference',ylabel='Frozen latent linear probe',title=name+' · '+drug)
    save(fig,'figure6_ce_scatter')
    fig,axs=plt.subplots(2,3,figsize=(14,7))
    for row,name in enumerate(MODEL_NAMES[1:]):
        d=np.load(OUT/f'dose_scaling_{filename(name)}.npz')
        for col,index in enumerate(d['window_indices']):
            ax=axs[row,col]; y=ref['y'][index,:,0]
            ax.plot(np.arange(1,31)/6,y,color='#222222',lw=2,label='Observed')
            for k,alpha in enumerate(d['alpha']): ax.plot(np.arange(1,31)/6,d['prediction'][k,col,:,0],label=f'α={alpha:g}',lw=1.2)
            ax.set(title=f'{name} · case {ref["case"][index]}',xlabel='Future minutes',ylabel='Predicted BIS')
    axs[0,0].legend(fontsize=7,frameon=False)
    fig.suptitle('Historical-dose scaling audit · not counterfactual treatment effects',fontsize=12)
    save(fig,'figure7_dose_scaling')
    render_large()

def render_large():
    f=pd.read_csv(OUT/'large_change_forecast_metrics.csv')
    t=pd.read_csv(OUT/'large_change_time_shift_metrics.csv')
    a=pd.read_csv(OUT/'large_change_action_ablation_metrics.csv')
    fig,axs=plt.subplots(1,3,figsize=(14,4.2))
    for name in ['Persistence']+MODEL_NAMES:
        d=f[(f.model==name)&(f.subset=='large_transition')&(f.target=='BIS')&(f.horizon_seconds>0)].sort_values('horizon_seconds')
        axs[0].plot(d.horizon_seconds/60,d.mae,'o-',label=name,color=COLORS[name])
    axs[0].set(xlabel='Forecast horizon (minutes)',ylabel='BIS MAE',title='Larger dose changes: 18,364 windows')
    axs[0].legend(frameon=False,fontsize=7)
    for name in MODEL_NAMES[1:]:
        d=t[(t.model==name)&(t.horizon_seconds==0)].sort_values('shift_seconds'); base=float(d[d.shift_seconds==0].mae.iloc[0])
        axs[1].plot(d.shift_seconds,d.normalized_degradation*100,'o-',label=name,color=COLORS[name])
        axs[1].fill_between(d.shift_seconds,d.delta_ci_low/base*100,d.delta_ci_high/base*100,color=COLORS[name],alpha=.15)
    axs[1].axhline(0,color='#555555',lw=.8); axs[1].set(xlabel='History shift Δ (seconds)',ylabel='Paired MAE increase (%)',title='Timing matters at larger changes')
    axs[1].set_xticks([-120,-60,0,60,120]); axs[1].legend(frameon=False,fontsize=7)
    for i,name in enumerate(MODEL_NAMES[1:]):
        for k,intervention in enumerate(['zero_action','matched_shuffle']):
            d=a[(a.model==name)&(a.subset=='large_transition')&(a.intervention==intervention)].iloc[0]
            x=i*3+k; v=100*d.normalized_degradation
            axs[2].bar(x,v,color=COLORS[name],alpha=1 if k==0 else .6)
            axs[2].errorbar(x,v,yerr=[[max(0,v-100*d.delta_ci_low/d.true_mae)],[max(0,100*d.delta_ci_high/d.true_mae-v)]],fmt='none',ecolor='black',capsize=3)
    axs[2].set_xticks([0,1,3,4],['T: zero','T: shuffle','R: zero','R: shuffle'],rotation=20)
    axs[2].set(ylabel='Paired MAE increase (%)',title='Removal and matched wrong actions')
    fig.suptitle('Supplementary event-magnitude audit · definition disclosed after initial test inspection',fontsize=11)
    save(fig,'figure8_large_change_audit')

def final_report(tables_,forecast,probe,timing,ablation,outcome,interpretation):
    ds=json.loads((OUT/'data_summary.json').read_text()); models=pd.read_csv(OUT/'model_summary.csv')
    checks=json.loads((OUT/'integrity_checks.json').read_text())
    matching=json.loads((OUT/'matched_shuffle_summary.json').read_text()); thresholds=json.loads((OUT/'event_thresholds.json').read_text())
    paired=pd.read_csv(OUT/'paired_forecast_comparisons.csv')
    large=json.loads((OUT/'large_change_protocol.json').read_text())
    lf=pd.read_csv(OUT/'large_change_paired_comparisons.csv')
    lt=pd.read_csv(OUT/'large_change_time_shift_metrics.csv').query('horizon_seconds == 0')
    title={'A':'Strong gap supported','B':'Partial gap','C':'Gap not supported'}[outcome]
    f=forecast[(forecast.subset=='overall')&(forecast.target=='BIS')]
    p=probe[probe.subset=='overall']
    t=timing[(timing.subset.isin(['overall','transition']))&(timing.horizon_seconds==0)]
    a=ablation[(ablation.subset.isin(['overall','transition']))&(ablation.horizon_seconds==0)]
    text=f'''# VitalDB intervention grounding diagnostic, seed 0

**Outcome {outcome} — {title}**

{interpretation}

## 1. Cohort construction and exact variables

Source: public VitalDB numeric per-track API. The installed official Python package is used for track-availability cohort queries. Current `load_case` reads full `.vital` containers, so it is intentionally avoided to honor the no-waveform requirement. Official references: [ppf_bis notebook](https://github.com/vitaldb/examples/blob/master/ppf_bis.ipynb), [API documentation](https://vitaldb.net/docs/?documentId=API%2FWeb_API_OpenDataset.md), [numeric track dictionary](https://physionet.org/files/vitaldb/1.0.0/track_names.csv). Copies, track IDs, metadata hashes and package versions are preserved.

Filters: age >18 (matching executable official example), weight >35 kg, General anesthesia, recorded case duration >7200 seconds, BIS, both administration tracks and both CE tracks. Complete plausible demographics are required. Seed 0 randomly selects at most 500 candidates before trajectory quality control. No first-BIS>=80 or last-BIS>=70 filter is used: these endpoint thresholds select clinical outcomes rather than missing-data quality and are unnecessary for history forecasting. Cases require >1 mL total recorded administration for each drug, positive reference CE and at least 30 usable windows.

Cohort counts: `{json.dumps(ds['cohort_stages'])}`. Selected candidates: **{ds['selected_before_qc']}**. Final eligible cases: **{ds['included_cases']}**, unique patients: **{ds['included_patients']}**. Excluded after selection: {ds['excluded_cases']}. Every exclusion and track choice is in `case_preprocessing_log.json`.

Track allowlist: BIS/BIS; Solar8000/HR; Solar8000/ART_MBP, FEM_MBP or NIBP_MBP; Orchestra/PPF20_RATE and RFTN20_RATE; fallback PPF20_VOL/RFTN20_VOL; and PPF20_CE/RFTN20_CE. RATE is mL/h and converted to mL per 10 seconds by dividing by 360; nominal solutions are propofol 20 mg/mL and remifentanil 20 µg/mL. RATE availability must exceed 90% of eligible cases; per-case valid coverage on BIS support must exceed 80%, otherwise volume differences are used. Fallback negative resets and >10 mL/10s increments are marked missing, not interpreted as true zero-dose stops. Actual action-track pairs: `{json.dumps(ds.get('actual_action_track_pairs',{}))}`. Actual MAP sources: `{json.dumps(ds.get('actual_map_sources',{}))}`. Data-derived exact usage is recorded per case. The RATE/volume semantics audit is saved in rate_volume_semantics_check.csv; it distinguishes large syringe resets from small negative volume corrections rather than summing only positive differences.

CE is a **device-computed TCI pharmacokinetic reference**, in µg/mL for propofol and ng/mL for remifentanil. It is not a measured concentration. CE, CP and target concentration are excluded from forecasting inputs and losses. Demographics: age, female indicator, weight and height.

## 2. Sampling, missingness and splits

10-second grid; latest valid numeric observation at or before each grid time, never a future observation; maximum carry-forward age 60 seconds. Longer and leading gaps remain missing, become training-mean zero after normalization, and receive explicit availability masks. Raw observed-bin BIS targets exclude forward-filled future labels. Valid ranges: BIS 1–100, HR 20–250/min, MAP 20–200 mmHg; direct RATE 0–3600 mL/h. Leading missing drug values are not treated as known no-infusion time.

History: 180 steps (30 minutes). Forecast: next 30 steps (5 minutes). Windows are generated dynamically at every 10-second anchor; no sliding-window tensors are persisted. A window requires >=80% BIS history coverage, >=95% history coverage for each action, >=90% observed future BIS coverage, current BIS and observed BIS at all four requested horizons. Arterial MAP is preferred when coverage is >=50%; cuff MAP may be a history covariate but is never a MAP target because repeated publication can represent a stale measurement.

Subject IDs, rather than case IDs alone, are shuffled with seed 0 and split approximately 70/15/15. All surgeries of one subject stay together. Exact IDs: `split_caseids.json` and `split_subjectids.json`. Split counts: `{json.dumps(ds['splits'])}`. All normalization, event thresholds, probe scaling and probe coefficients are derived from training patients only.

## 3. Models and optimization

A/B share a two-layer, width-64, four-head Transformer encoder and a 30-step direct prediction head. Only B receives drug history. C is a compact **deterministic RSSM-style** baseline: two-layer GRU history encoder, 64-dimensional latent, GRUCell action-conditioned future transition, and physiology decoder. It is not a stochastic variational RSSM. Its future rollout holds the last observed action constant; no model receives actual future actions. All have residual prediction from current physiology. Standardized BIS MSE plus 0.25 standardized arterial-MAP MSE trains the entire forecast model. No CE supervision or pretraining is used.

Fixed AdamW settings in config.yaml; at most 20 epochs; up to 80,000 randomly sampled distinct training windows per epoch; validation uses a fixed 60-second stride, early stopping patience 5 and lowest validation BIS MAE checkpoint. Test and probe extraction use all valid 10-second windows. One seed, no hyperparameter search.

{markdown(models)}

## 4. Forecasting results

Point-horizon errors measure precisely t+30,60,180,300 seconds. The full-trajectory error averages all observed points in the next five minutes. Primary aggregation weights patients equally, first averaging windows within patient; RMSE is the square root of patient-weighted MSE. Paired uncertainty uses 1,000 patient-cluster bootstrap replicates, preserving serial and cross-case dependence within patients. Pooled-window MAE is also saved. Persistence repeats current physiology and tests whether 'good forecasting' exceeds an autoregressive shortcut.

{markdown(tables_['overall'])}

Timing column is max absolute full-trajectory TD across the prescribed shifts. ARD/MWAD are absolute BIS MAE increases, averaged over five minutes; MWAD uses the matched subset and its own paired true-action error.

{markdown(f[['model','horizon_label','mae','mae_ci_low','mae_ci_high','rmse','n_windows','n_patients']])}

Paired full-trajectory comparisons:

{markdown(paired[(paired.subset.isin(['overall','transition']))&(paired.horizon_seconds==0)])}

![Forecasting](../plots/figure1_forecasting.png)

## 5. Frozen representation probes

Entire forecasting checkpoints are frozen. A separate ridge linear probe (alpha=100; train-only feature standardization) is fit to each CE using **all valid training windows** with equal total patient weight. No encoder update or test CE fitting occurs. Required controls use only current BIS/HR/MAP with availability flags, or the uncompressed 180x2 raw-dose history with two missing-fraction indicators. Supplemental controls add the same four demographics to each of these inputs, checking whether apparent latent information merely reflects patient characteristics available to the forecasting model. No CE appears in any control input. R² and Pearson use pooled moments with equal patient mass; patient bootstrap supplies confidence intervals. Negative R² values are retained.

{markdown(p[['representation','drug','r2','r2_ci_low','r2_ci_high','mae','pearson','n_patients']])}

![Probes](../plots/figure2_ce_probes.png)
![Reference scatter](../plots/figure6_ce_scatter.png)

## 6. Removal and matched wrong actions

Zero-action sets physical doses to zero while keeping masks. Matched shuffle uses another test **patient**, requires each current BIS/HR/MAP difference <=0.5 training SD and action-history RMS difference >=0.25 training dose SD, and searches the 128 nearest physiological candidates. No-match windows are excluded only from shuffle comparisons. All test CE and future outcomes are excluded from matching. Match coverage and balance: `{json.dumps(matching)}`.

{markdown(a[['model','intervention','subset','true_mae','perturbed_mae','degradation','delta_ci_low','delta_ci_high','n_windows']])}

## 7. Timing perturbation

Only historical actions are shifted simultaneously. Positive Δ delays them, negative Δ advances them within the observed history. Edge replication pads unavailable shifted regions; there is no wrap and no reading beyond the forecast anchor. Associated action availability masks move with their samples. Physics history, target, demographics and weights remain fixed. Dose perturbation size is reported because a flat curve during constant infusion carries little diagnostic information.

{markdown(t[['model','subset','shift_seconds','mae','rmse','delta_mae','delta_ci_low','delta_ci_high','normalized_degradation','action_mean_abs_change_ml_per_10s']])}

![Timing curve](../plots/figure3_timing_sensitivity.png)

## 8. Transition-event evaluation

For each drug, compare adjacent one-minute median administration windows, requiring all 12 samples to be valid. 'On' is the training positive-dose 10th percentile; substantial change is the training 75th percentile of nonzero absolute median changes. Thresholds (mL/10s): `{json.dumps(thresholds)}`. Initiation requires a zero pre-median and a post-median above the on threshold; stop requires the reverse. Numerical zero tolerance is 1e-6 mL/10s. Ongoing increases/decreases require two positive medians and cross the magnitude threshold. This pre-evaluation clarification is documented in protocol_amendments.md. Detections for one drug are at least two minutes apart. Evaluation anchors are from confirmation through the next two minutes, so the observed change is in history. Initial induction before a full 30-minute history is not represented; later initiations/restarts are. These are observational pump-record changes, not randomized interventions. Separate event-type counts/errors, transition probes and timing curves are saved.

{markdown(tables_['transition'])}

{markdown(probe[(probe.subset=='transition')][['representation','drug','r2','mae','pearson','n_windows','n_patients']])}

![Transitions](../plots/figure4_transition_audit.png)

### 8.1. Supplementary larger-change audit

Inspecting the initial protocol revealed that its 75th-percentile nonzero one-minute changes were only 0.3515 and 0.3390 mL/h. They include many small automatic pump adjustments and should not by themselves be described as substantial dose changes. After the initial Transformer timing results were inspected, a supplementary definition was added using only TRAIN administration values: the larger-change threshold is the maximum of the original threshold and half the IQR of positive administration rates. This yields **4.402 mL/h propofol and 6.668 mL/h remifentanil**. Initiation and stop retain their genuine zero-crossing definitions. The original analysis is retained in every result file; rows with subset `large_transition` and the `large_change_*` files contain this supplementary audit. This is an explicitly exploratory protocol extension, not a preregistered confirmatory result. No model, normalization, checkpoint or probe was refit.

Larger-change subset: **{large['n_test_windows']} windows**, **{large['n_test_cases']} cases**, **{large['n_test_patients']} patients**. Window counts by type (0 excluded, 1 initiation, 2 increase, 3 decrease, 4 stop): `{json.dumps(large['kind_window_counts'])}`. See `large_change_protocol.json` for exact training quantiles, timestamp and the frozen definition. The transition files include both original and larger-event results.

{markdown(tables_['large_transition'])}

{markdown(lf)}

{markdown(lt[['model','shift_seconds','mae','delta_mae','delta_ci_low','delta_ci_high','normalized_degradation']])}

![Larger changes](../plots/figure8_large_change_audit.png)

## 9. Representative patients and dose scaling

The three patients are fixed at the 15th, 50th and 85th percentiles of sorted test case IDs, independent of model errors. Trajectory plots compare observed BIS to 5-minute-ahead predictions, pump rates, TCI CE and frozen-latent CE predictions. Dose-scaling anchors use the middle transition window when available, otherwise the middle usable window. Scaling historical administration by 0, 0.5, 1 and 1.5 is an input-response audit; it does **not** estimate counterfactual treatment effects or require universal monotonicity.

![Patients](../plots/figure5_patient_trajectories.png)
![Scaling](../plots/figure7_dose_scaling.png)

## 10. Integrity, interpretation and limits

Focused checks: `{json.dumps(checks)}`. All three models use identical cases and usable anchor rules. Predictions, checkpoints, per-patient matching, configuration and numeric source provenance are retained. No waveform or full-case container was downloaded.

The preregistered diagnostic thresholds are descriptive, not clinical standards. Good forecasting requires full-trajectory BIS MAE <=5 and >=5% improvement on persistence. Meaningful action gain requires >=5% improvement on state-only with paired CI excluding zero. Strong CE recovery requires both R²>=0.5 and >=0.1 above instantaneous physiology; near-flat timing means all |TD|<=2%. Those thresholds cannot establish physical or causal fidelity. Linear probes test readily decodable TCI-reference information, not all information in a nonlinear latent. Historical-input perturbations condition on physiology that already reflects past drug exposure; predictive insensitivity does not uniquely prove absence of a drug mechanism. TCI-controlled rates, unobserved surgery/other drugs, one center, one seed, a short horizon, unknown future actions, overlapping windows and a deterministic compact RSSM limit generalization. A flat timing curve must be considered together with perturbation magnitude and dose-change results.

**Final evidence classification: Outcome {outcome} — {title}.**
'''
    (OUT/'FINAL_REPORT.md').write_text(text,encoding='utf-8')

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--outcome',choices=['A','B','C']); parser.add_argument('--interpretation-file'); parser.add_argument('--no-plots',action='store_true')
    args=parser.parse_args()
    forecast=pd.read_csv(OUT/'forecast_metrics.csv'); probe=pd.read_csv(OUT/'ce_probe_metrics.csv')
    timing=pd.read_csv(OUT/'time_shift_metrics.csv'); ablation=pd.read_csv(OUT/'action_ablation_metrics.csv')
    tab=tables(forecast,probe,timing,ablation)
    if not args.no_plots: render(forecast,probe,timing,ablation)
    if args.outcome:
        if not args.interpretation_file: raise ValueError('Supply evidence-based interpretation after inspecting results')
        final_report(tab,forecast,probe,timing,ablation,args.outcome,Path(args.interpretation_file).read_text())
    print('REPORT_ARTIFACTS_READY',flush=True)

if __name__=='__main__': main()
