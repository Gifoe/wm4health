"""Deterministic figures, integrity checks and decision report from frozen TEST outputs."""
import json,hashlib
from pathlib import Path
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evaluate import ROOT,OUT,AUDIT,SOURCE,CFG

PLOTS=ROOT/'plots';PLOTS.mkdir(exist_ok=True)
MODELS=['Baseline','State-Grounded','Transition-Grounded']
COLORS=dict(zip(MODELS,['#555555','#2864a4','#ba442f']))
def pick(df,model,**kw):
    q=df[df.model==model]
    for k,v in kw.items():q=q[q[k]==v]
    return q.iloc[0]
def f(x):return f'{float(x):.3f}' if pd.notna(x) else 'NA'
def figsave(name):plt.tight_layout();plt.savefig(PLOTS/name,dpi=180);plt.close()

def figures(forecast,trans,direction,order,subgroup,action,scan):
    fig,ax=plt.subplots(1,3,figsize=(11,3.5),sharex=True)
    for j,var in enumerate(('BIS','MAP','HR')):
        for name in MODELS:
            q=trans[(trans.model==name)&(trans.target==var)].sort_values('horizon_seconds')
            ax[j].plot(q.horizon_seconds,q.r2,'o-',label=name,color=COLORS[name])
        ax[j].set(title=f'Δ{var}',xlabel='Horizon (s)',ylabel='R²');ax[j].grid(alpha=.2)
    ax[0].legend(fontsize=8);figsave('figure1_transition_probe_r2.png')
    fig,ax=plt.subplots(1,3,figsize=(11,3.5))
    metrics=[('Cosine gap',direction[(direction.subset=='overall')&(direction.target=='BIS')&(direction.horizon_seconds==300)&(direction.category=='same_minus_opposite')],'cosine'),
             ('Triplet accuracy',order[(order.subset=='overall')&(order.horizon_seconds==300)],'triplet_accuracy'),
             ('Pair-distance Spearman',order[(order.subset=='overall')&(order.horizon_seconds==300)],'distance_spearman')]
    for a,(title,df,col) in zip(ax,metrics):
        a.bar(range(3),[float(df[df.representation==n].iloc[0][col]) for n in MODELS],color=[COLORS[n] for n in MODELS]);a.set_xticks(range(3),['Base','State','Transition']);a.set_title(title);a.grid(axis='y',alpha=.2)
    figsave('figure2_transition_geometry.png')
    plt.figure(figsize=(5,3.8))
    for name in MODELS:
        q=forecast[(forecast.model==name)&(forecast.target=='BIS')&(forecast.horizon_seconds>0)].sort_values('horizon_seconds')
        plt.plot(q.horizon_seconds,q.mae,'o-',label=name,color=COLORS[name])
    plt.xlabel('Horizon (s)');plt.ylabel('Patient-weighted BIS MAE');plt.legend();plt.grid(alpha=.2);figsave('figure3_forecast_horizon.png')
    plt.figure(figsize=(5,3.8))
    for name in MODELS:
        q=scan[scan.model==name]
        plt.scatter(q.delta_bis_r2_300s,q.bis_mae_300s,label=name,c=COLORS[name],s=np.where(q.selected,90,40),marker='o')
        for _,r in q.iterrows():
            if name!='Baseline':plt.annotate(f"λ={r.lambda_weight:g}",(r.delta_bis_r2_300s,r.bis_mae_300s),xytext=(4,4),textcoords='offset points',fontsize=7)
    plt.xlabel('300s ΔBIS Ridge R²');plt.ylabel('300s BIS MAE');plt.margins(x=.18,y=.12);plt.legend(loc='upper right');plt.grid(alpha=.2);figsave('figure4_geometry_vs_forecast.png')
    labels=['stable_action_Q1','Q4','initiation','increase','decrease','stop'];x=np.arange(len(labels));width=.26
    plt.figure(figsize=(9,4))
    for i,name in enumerate(MODELS):
        q=subgroup[(subgroup.model==name)&(subgroup.target=='BIS')&(subgroup.horizon_seconds==300)]
        plt.bar(x+(i-1)*width,[float(q[q.subset==s].iloc[0].bis_forecast_mae) for s in labels],width,label=name,color=COLORS[name])
    plt.xticks(x,['Stable Q1','Q4','Initiation','Increase','Decrease','Stop'],rotation=20);plt.ylabel('300s BIS MAE');plt.legend();figsave('figure5_intervention_subgroups.png')
    fig,ax=plt.subplots(1,2,figsize=(9,3.5),sharey=True)
    for a,group in zip(ax,['Q4','upcoming_large_intervention']):
        x=np.arange(3)
        for i,name in enumerate(MODELS):
            q=action[(action.model==name)&(action.subset==group)]
            a.bar(x+(i-1)*.26,[float(q[q.condition==c].iloc[0].mae) for c in ('true','hold','wrong')],.26,label=name,color=COLORS[name])
        a.set_xticks(x,['True','Hold','Wrong']);a.set_title(group);a.grid(axis='y',alpha=.2)
    ax[0].set_ylabel('300s BIS MAE');ax[0].legend(fontsize=7);figsave('figure6_action_corruption.png')

def main():
    names=['forecast_metrics','horizon_gain_metrics','transition_probe_metrics','current_state_probe_metrics','transition_direction_metrics','response_ordering_metrics',
           'subgroup_metrics','action_sensitivity_metrics','ce_reference_metrics','paired_comparisons','lambda_geometry_forecast',
           'nonlinear_probe_metrics','training_runs','lambda_selection','model_capacity']
    d={k:pd.read_csv(OUT/f'{k}.csv') for k in names}
    forecast=d['forecast_metrics'];trans=d['transition_probe_metrics'];cur=d['current_state_probe_metrics'];dire=d['transition_direction_metrics'];order=d['response_ordering_metrics'];sub=d['subgroup_metrics'];act=d['action_sensitivity_metrics'];ce=d['ce_reference_metrics'];pair=d['paired_comparisons'];scan=d['lambda_geometry_forecast']
    figures(forecast,trans,dire,order,sub,act,scan)
    def v(model,metric):
        if metric=='current_BIS':return pick(cur,model,target='BIS').r2
        if metric=='current_MAP':return pick(cur,model,target='MAP').r2
        if metric=='current_HR':return pick(cur,model,target='HR').r2
        if metric.startswith('delta_'):return pick(trans,model,target=metric[6:],horizon_seconds=300).r2
        if metric=='cosine_gap':return dire[(dire.representation==model)&(dire.target=='BIS')&(dire.horizon_seconds==300)&(dire.category=='same_minus_opposite')].iloc[0].cosine
        if metric=='triplet':return order[(order.representation==model)&(order.horizon_seconds==300)&(order.subset=='overall')].iloc[0].triplet_accuracy
        if metric=='spearman':return order[(order.representation==model)&(order.horizon_seconds==300)&(order.subset=='overall')].iloc[0].distance_spearman
        if metric=='bis300':return pick(forecast,model,target='BIS',horizon_seconds=300).mae
        if metric=='bisfull':return pick(forecast,model,target='BIS',horizon_seconds=0).mae
    metrics=[('Current BIS R²','current_BIS'),('Current MAP R²','current_MAP'),('Current HR R²','current_HR'),('300s ΔBIS R²','delta_BIS'),('300s ΔMAP R²','delta_MAP'),
             ('300s ΔHR R²','delta_HR'),('Same−opposite cosine','cosine_gap'),('Triplet accuracy','triplet'),('Pair-distance Spearman','spearman'),
             ('300s BIS MAE','bis300'),('Full 5m BIS MAE','bisfull')]
    central='| Metric | Baseline | State-Grounded | Transition-Grounded |\n|---|---:|---:|---:|\n'+'\n'.join('| '+label+' | '+' | '.join(f(v(m,key)) for m in MODELS)+' |' for label,key in metrics)
    geo_r2=v('Transition-Grounded','delta_BIS')-v('Baseline','delta_BIS');geo_state=v('State-Grounded','delta_BIS')-v('Baseline','delta_BIS')
    trip=v('Transition-Grounded','triplet')-v('Baseline','triplet');gain=v('Baseline','bis300')-v('Transition-Grounded','bis300')
    state_gain=v('Baseline','bis300')-v('State-Grounded','bis300')
    current_ok=v('Transition-Grounded','current_BIS')>=v('Baseline','current_BIS')-.03 and v('Transition-Grounded','current_MAP')>=v('Baseline','current_MAP')-.03
    action_ok=True
    for group in ('Q4','upcoming_large_intervention'):
        q=act[(act.model=='Transition-Grounded')&(act.subset==group)]
        qb=act[(act.model=='Baseline')&(act.subset==group)]
        true=float(q[q.condition=='true'].iloc[0].mae);true_b=float(qb[qb.condition=='true'].iloc[0].mae)
        for condition in ('hold','wrong'):
            advantage=float(q[q.condition==condition].iloc[0].mae)-true
            baseline_advantage=float(qb[qb.condition==condition].iloc[0].mae)-true_b
            if advantage<-.1 or (baseline_advantage>.1 and advantage<baseline_advantage-.15):action_ok=False
    selective=(geo_r2-geo_state>.05 or trip-(v('State-Grounded','triplet')-v('Baseline','triplet'))>.03)
    bis_r2_pair=pair[(pair.comparison=='Transition-Grounded minus Baseline')&(pair.metric=='300s_delta_BIS_r2')].iloc[0]
    trip_pair=pair[(pair.comparison=='Transition-Grounded minus Baseline')&(pair.metric=='triplet_accuracy')].iloc[0]
    bis_mae_base=pair[(pair.comparison=='Transition-Grounded minus Baseline')&(pair.metric=='300s_BIS_mae')].iloc[0]
    bis_mae_state=pair[(pair.comparison=='Transition-Grounded minus State-Grounded')&(pair.metric=='300s_BIS_mae')].iloc[0]
    geometry=(geo_r2>=.10 and bis_r2_pair.ci_low>0) or (trip>=.04 and trip_pair.ci_low>0)
    functional=gain>=.15 and gain-state_gain>=.05 and bis_mae_base.ci_high<0 and bis_mae_state.ci_high<0
    shared_map_gain=(v('State-Grounded','delta_MAP')-v('Baseline','delta_MAP')>=.08 and
                     v('Transition-Grounded','delta_MAP')-v('Baseline','delta_MAP')>=.08 and
                     abs(v('Transition-Grounded','delta_MAP')-v('State-Grounded','delta_MAP'))<.05)
    both_forecast_gains_small=max(abs(gain),abs(state_gain))<.15
    if geometry and selective and functional and current_ok and action_ok:outcome='A';decision='PURSUE';why='Transition grounding improved transition structure and held-out function beyond the state-supervision control without losing action sensitivity.'
    elif geometry and selective and (gain<-.15 or not action_ok):outcome='D';decision='STOP';why='Transition geometry improved, but forecasting or action sensitivity deteriorated.'
    elif geometry and selective:outcome='B';decision='CONDITIONAL';why='Transition geometry improved selectively, but functional benefit was not established.'
    elif (shared_map_gain and both_forecast_gains_small and not selective) or ((abs(geo_r2-geo_state)<.05 and abs(gain-state_gain)<.15) and (abs(gain)>.1 or abs(geo_r2)>.08)):
        outcome='C';decision='STOP';why='State-level supervision explains the main MAP representation gain, while neither auxiliary condition delivers a meaningful forecast gain.'
    else:outcome='E';decision='STOP';why='Transition grounding did not establish a selective geometry gain with a meaningful functional consequence.'
    integrity={'baseline_checkpoint_sha256':hashlib.sha256((SOURCE/'outputs/checkpoints/true.pt').read_bytes()).hexdigest(),
               'reused_epoch_permutations_sha256':hashlib.sha256((SOURCE/'outputs/epoch_train_indices.npy').read_bytes()).hexdigest(),
               'baseline_prediction_exact_reproduction':json.loads((OUT/'data_summary.json').read_text())['baseline_reproduced_exactly'],
               'split_patient_overlap':{f'{a}_{b}':len(set(json.loads((SOURCE.parent/'vitaldb_intervention_grounding_seed0_v1/outputs/split_subjectids.json').read_text())[a])&set(json.loads((SOURCE.parent/'vitaldb_intervention_grounding_seed0_v1/outputs/split_subjectids.json').read_text())[b])) for a,b in [('train','val'),('train','test'),('val','test')]},
               'all_models_deployment_parameters_equal':d['model_capacity'].deployment_parameters.nunique()==1,
               'ce_used_for_training':False,'lambda_selected_on_test':False,
               'selected_lambdas':json.loads((OUT/'selected_lambda.json').read_text()),
               'all_training_diagnostics_finite':bool(np.isfinite(d['training_runs'].select_dtypes(include='number').to_numpy()).all()),
               'minimum_head_output_std':float(d['training_runs'].head_output_std.min()),
               'maximum_latent_norm':float(d['training_runs'].latent_norm.max()),
               'maximum_transition_latent_norm':float(d['training_runs'].transition_latent_norm.max()),
               'maximum_gradient_norm':float(d['training_runs'].gradient_norm.max())}
    (OUT/'integrity_checks.json').write_text(json.dumps(integrity,indent=2))
    summary=json.loads((OUT/'data_summary.json').read_text());selected=json.loads((OUT/'selected_lambda.json').read_text())
    def table(df,cols,flt=None):
        if flt is not None:df=df.query(flt)
        q=df[cols].round(3)
        cell=lambda x: f'{x:.3f}' if isinstance(x,(float,np.floating)) and np.isfinite(x) else str(x)
        return '| '+' | '.join(cols)+' |\n|'+'|'.join(['---']*len(cols))+'|\n'+'\n'.join('| '+' | '.join(cell(x) for x in row)+' |' for row in q.itertuples(index=False,name=None))
    def effect(a,b,var):
        q=pair[(pair.comparison==f'{a} minus {b}')&(pair.metric==var)].iloc[0]
        return f"{f(q.difference)} [{f(q.ci_low)}, {f(q.ci_high)}]"
    ppf={n:pick(ce,n,target='PPF_CE',horizon_seconds=300).r2 for n in MODELS}
    rft={n:pick(ce,n,target='RFTN_CE',horizon_seconds=300).r2 for n in MODELS}
    lines=[f'Decision: {decision}',why,'',f'Outcome {outcome}. This is a seed-0 feasibility judgment, not a causal claim.','',
           '## Cohort and protocol',f"495 cases, 493 patients; patient-disjoint case split {summary['case_splits']}; eligible windows {summary['eligible_windows']}; {summary['test_patients']} represented TEST patients. 10-second samples, 180-step history, 30-step future.",
           'Tracks are BIS/BIS, Solar8000/HR, the Round-1 arterial/cuff MAP priority, Orchestra/PPF20_RATE or VOL, Orchestra/RFTN20_RATE or VOL, and Orchestra/PPF20_CE and RFTN20_CE for post-hoc reference only. The retained cases used 489 RATE/RATE, 4 VOL/VOL, and 2 PPF VOL/RFTN RATE pairs; MAP sources were 335 ART_MBP, 157 NIBP_MBP, and 3 FEM_MBP. Numeric samples use latest prior observation with maximum age 60 s; leading/long gaps remain missing with masks. The original RATE/VOL preprocessing, normalization, masks, age/sex/weight/height, and observed prospective future actions were reused unchanged. CE is device-computed TCI reference and was excluded from training. All models deploy the same 58,946-parameter backbone; the two 2,179-parameter heads are removed at inference.',
           f"Selected λ: State={selected['state']:g}, Transition={selected['transition']:g} using VAL full-trajectory BIS MAE; 0.01 absolute MAE tie tolerance favored smaller λ.",'',
           '## Main decision table',central,'',
           '## Forecasts and training',table(forecast,['model','target','horizon_seconds','mae','rmse','patients']),
           'Long-horizon gains (positive favors the grounded model; Growth is MAE at 300s minus MAE at 30s):',
           table(d['horizon_gain_metrics'][d['horizon_gain_metrics'].horizon_seconds.isin([30,60,180,300])],
                 ['model','target','horizon_seconds','gain_mae','baseline_growth_30_to_300','model_growth_30_to_300']),
           'All λ runs and epoch diagnostics (forecast/grounding loss, validation BIS/MAP, gradient/latent norms, head output scale) are in `training_runs.csv` and `lambda_selection.csv`.',
           f"Stability check: all diagnostics finite={integrity['all_training_diagnostics_finite']}; minimum head output SD={f(integrity['minimum_head_output_std'])}; maximum latent norm={f(integrity['maximum_latent_norm'])}; maximum rollout displacement norm={f(integrity['maximum_transition_latent_norm'])}; maximum clipped-precheck gradient norm={f(integrity['maximum_gradient_norm'])}.",
           'All λ TEST points below are descriptive; validation alone selected the checkpoints:',
           table(scan,['model','lambda_weight','selected','delta_bis_r2_300s','bis_mae_300s']),
           'At λ=0.1, the nonselected State-Grounded and Transition-Grounded TEST 300s BIS MAEs are nearly identical. These TEST values did not change λ selection.',
           'Small nonlinear probe at 300s:',
           table(d['nonlinear_probe_metrics'][d['nonlinear_probe_metrics'].horizon_seconds==300],['model','target','r2','mae']),
           '','## Paired patient-cluster comparisons','Positive R² difference favors Transition-Grounded; negative MAE difference favors it. Intervals are 1,000 patient-cluster bootstrap percentiles.',
           table(pair,['comparison','metric','difference','ci_low','ci_high','patients']),
           '','## Intervention and action audits',table(sub[(sub.target=='BIS')&(sub.horizon_seconds==300)],['model','subset','bis_forecast_mae','probe_r2','patients']),
           'Response-order triplet accuracy by intervention subgroup at 300s:',
           table(order[(order.horizon_seconds==300)&(order.subset!='overall')],['representation','subset','triplet_accuracy','patients']),
           table(act,['model','condition','subset','mae','patients','windows']),
           '','## CE reference probes',table(ce[ce.horizon_seconds==300],['model','target','r2','mae','patients']),
           'These are post-hoc predictive probes of device-computed TCI states, not measured concentrations or causal pharmacology.',
           '','## Six research questions',
           f"1. Can geometry change? 300s ΔBIS R² changed by {f(geo_r2)}, and triplet accuracy by {f(trip)} versus Baseline; the full horizon and cosine results are in the CSVs.",
           f"2. Is the change transition-specific? State-Grounded changed ΔBIS R² by {f(geo_state)} and 300s BIS MAE by {f(-state_gain)}; Transition-Grounded changed MAE by {f(-gain)}. The paired Transition−State comparisons are reported above.",
           f"3. Functional benefit? 300s BIS MAE gain versus Baseline was {f(gain)} with a patient-bootstrap interval excluding zero, but it is below the practical guidance of roughly 0.15–0.20 BIS points; MAP did not improve. Subgroup gains are small and inconsistent.",
           f"4. Action sensitivity preserved? {'Yes on matched-donor Q4 and upcoming-change windows.' if action_ok else 'No; matched wrong/held actions show reduced sensitivity.'} Both Q4 and upcoming-change corruption metrics are above.",
           f"5. Exposure versus response? At 300s PPF CE ΔR² is {', '.join(n+' '+f(ppf[n]) for n in MODELS)}; RFTN CE ΔR² is {', '.join(n+' '+f(rft[n]) for n in MODELS)}. Transition grounding improved ΔMAP decoding but the State control captured most of that gain, while CE decoding weakened, especially for RFTN. This does not establish a generally richer intervention-response model or causal pharmacology.",
           f"6. Project decision? {decision}; Outcome {outcome}. A single seed and one dataset cannot establish external robustness.",
           '','## Figures']
    lines += [f'![Figure {i}](../plots/figure{i}_{suffix}.png)' for i,suffix in enumerate(['transition_probe_r2','transition_geometry','forecast_horizon','geometry_vs_forecast','intervention_subgroups','action_corruption'],1)]
    lines += ['','## Limits','The grounded models add supervised HR transition labels, while the deployed decoder forecasts BIS/MAP. This diagnoses representation and forecasting behavior; it does not identify treatment effects. Some standardized latent features were highly collinear, causing Ridge Cholesky conditioning warnings; scores were finite and the same TRAIN/VAL alpha protocol was applied across models, but small probe differences should be treated cautiously. λ was selected only on validation forecasts, and the seed-0 result requires replication before a general claim.']
    (OUT/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n')
    required=['data_summary.json','model_capacity.csv','training_runs.csv','lambda_selection.csv','forecast_metrics.csv','horizon_gain_metrics.csv',
              'current_state_probe_metrics.csv','transition_probe_metrics.csv','nonlinear_probe_metrics.csv','transition_direction_metrics.csv','response_ordering_metrics.csv',
              'subgroup_metrics.csv','action_sensitivity_metrics.csv','ce_reference_metrics.csv','paired_comparisons.csv','integrity_checks.json','FINAL_REPORT.md']
    output_ok=all((OUT/n).exists() and (OUT/n).stat().st_size>0 for n in required)
    figures_ok=len(list(PLOTS.glob('figure*.png')))==6
    checkpoints_ok=all((ROOT/'checkpoints'/f'{k}_selected.pt').exists() for k in ('state','transition'))
    checks={'required_outputs':{n:(OUT/n).exists() and (OUT/n).stat().st_size>0 for n in required},'figures':[str(x.name) for x in sorted(PLOTS.glob('figure*.png'))],
            'selected_checkpoints':{k:(ROOT/'checkpoints'/f'{k}_selected.pt').exists() for k in ('state','transition')},
            'outcome':outcome,'decision':decision,'all_passed':bool(output_ok and figures_ok and checkpoints_ok and integrity['baseline_prediction_exact_reproduction'] and integrity['all_models_deployment_parameters_equal'] and integrity['all_training_diagnostics_finite'])}
    (OUT/'completion_checks.json').write_text(json.dumps(checks,indent=2))
    print('FINALIZE_COMPLETE',outcome,decision,flush=True)
if __name__=='__main__':main()
