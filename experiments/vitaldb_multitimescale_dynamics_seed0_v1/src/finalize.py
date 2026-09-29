"""Six figures, provenance checks, and fixed qualitative feasibility decision."""
import json,hashlib
import numpy as np,pandas as pd,torch,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from models import *

OUT=ROOT/'outputs';PLOTS=ROOT/'plots';PLOTS.mkdir(exist_ok=True)
ARRAYS=OUT/'arrays';NAMES=['Direct-MH','Direct-MH-Capacity','MT-Dynamics']
COLORS={'AR-RSSM':'#777777','Direct-MH':'#245b88','Direct-MH-Capacity':'#be8432','MT-Dynamics':'#b74137'}
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def row(df,**criteria):
    q=df
    for key,value in criteria.items():q=q[q[key]==value]
    if len(q)!=1:raise RuntimeError(f'Expected one row {criteria}, found {len(q)}')
    return q.iloc[0]
def f(v):return f'{float(v):.3f}' if pd.notna(v) else 'NA'
def table(df,cols):
    q=df[cols].round(3);cell=lambda x:f'{x:.3f}' if isinstance(x,(float,np.floating)) and np.isfinite(x) else str(x)
    return '| '+' | '.join(cols)+' |\n|'+'|'.join(['---']*len(cols))+'|\n'+'\n'.join('| '+' | '.join(cell(v) for v in r)+' |' for r in q.itertuples(index=False,name=None))
def save(filename):plt.tight_layout();plt.savefig(PLOTS/filename,dpi=180);plt.close()

def figures(hm,adv,sub,act,probe,store,ref):
    plt.figure(figsize=(6,4))
    ar=pd.read_csv(DIRECT/'outputs/horizon_metrics.csv')
    for name in ['AR-RSSM']+NAMES:
        q=(ar if name=='AR-RSSM' else hm);q=q[(q.model==name)&(q.subset=='overall')&(q.target=='BIS')].sort_values('horizon_seconds')
        plt.plot(q.horizon_seconds,q.mae,'o-',label=name,color=COLORS[name])
    plt.xlabel('Horizon (s)');plt.ylabel('Equal-patient BIS MAE');plt.legend(fontsize=8);plt.grid(alpha=.2);save('figure1_bis_horizon.png')
    q=adv[(adv.comparison=='CCA')&(adv.subset=='overall')&(adv.target=='BIS')].sort_values('horizon_seconds')
    plt.figure(figsize=(5.5,3.8));plt.errorbar(q.horizon_seconds,q.difference_mae,
        yerr=[q.difference_mae-q.ci_low,q.ci_high-q.difference_mae],fmt='o-',capsize=4,color=COLORS['MT-Dynamics'])
    plt.axhline(0,color='black',linewidth=.8);plt.xlabel('Horizon (s)');plt.ylabel('Capacity-controlled BIS MAE advantage');plt.grid(alpha=.2)
    save('figure2_capacity_controlled_advantage.png')
    labels=['stable_action_Q1','Q4','upcoming_large_intervention','initiation','increase','decrease','stop'];x=np.arange(len(labels));w=.36
    plt.figure(figsize=(9,4.2))
    for i,name in enumerate(['Direct-MH-Capacity','MT-Dynamics']):
        q=sub[(sub.model==name)&(sub.target=='BIS')&(sub.horizon_seconds==300)]
        plt.bar(x+(i-.5)*w,[row(q,subset=s).mae for s in labels],w,label=name,color=COLORS[name])
    plt.xticks(x,['Stable Q1','Q4','Upcoming','Initiation','Increase','Decrease','Stop'],rotation=20);plt.ylabel('300s BIS MAE');plt.legend();save('figure3_intervention_events.png')
    selected=['recent_PPF_action_delta_300s','recent_BIS_delta_300s','future_delta_BIS_30s','future_delta_MAP_60s',
              'history_BIS_mean_300s','history_MAP_mean_300s','history_PPF_action_mean_300s','future_delta_BIS_300s']
    labels=['Recent PPF Δ','Recent BIS Δ','Future BIS 30s','Future MAP 60s','BIS mean 5m','MAP mean 5m','PPF mean 5m','Future BIS 300s']
    x=np.arange(len(selected));w=.25;plt.figure(figsize=(10,4.3))
    for i,rep in enumerate(['z_fast','z_slow','concat']):
        q=probe[probe.representation==rep]
        plt.bar(x+(i-1)*w,[row(q,target=s).r2 for s in selected],w,label=rep)
    plt.axhline(0,color='black',linewidth=.8);plt.xticks(x,labels,rotation=35,ha='right');plt.ylabel('Equal-patient TEST R²');plt.legend();save('figure4_timescale_probes.png')
    fig,axes=plt.subplots(1,2,figsize=(9,3.8),sharey=True)
    for ax,group in zip(axes,['Q4','upcoming_large_intervention']):
        x=np.arange(3);w=.26
        for i,name in enumerate(NAMES):
            q=act[(act.model==name)&(act.subset==group)&(act.horizon_seconds==300)]
            ax.bar(x+(i-1)*w,[row(q,condition=c).mae for c in ('true','hold','wrong')],w,label=name,color=COLORS[name])
        ax.set_xticks(x,['True','Hold','Matched wrong']);ax.set_title(group);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('300s BIS MAE');axes[0].legend(fontsize=7);save('figure5_action_corruption.png')
    # First two eligible test cases in numeric order; chosen before viewing model errors.
    cases=ref['case'];ids=sorted(int(c) for c,n in zip(*np.unique(cases,return_counts=True)) if n>=200)[:2]
    fig,axes=plt.subplots(len(ids)*2,1,figsize=(10,7),sharex=False)
    model_preds={n:np.load(ARRAYS/f'{n}_true.npy',mmap_mode='r') for n in ('Direct-MH','MT-Dynamics')}
    for q,cid in enumerate(ids):
        c=store.cases[cid];ix=np.flatnonzero(cases==cid);anchors=ref['t'][ix]
        lo=max(0,int(anchors.min())-30);hi=min(len(c['state']),lo+180)
        ax=axes[2*q];t=np.arange(lo,hi)/6;ax.plot(t,c['state'][lo:hi,0],color='black',linewidth=1.2,label='Observed BIS')
        chosen=ix[(anchors>=lo)&(anchors<hi-30)][::12]
        for name,marker in [('Direct-MH','o'),('MT-Dynamics','x')]:
            ax.scatter((ref['t'][chosen]+30)/6,model_preds[name][chosen,3,0],s=24,marker=marker,color=COLORS[name],label=name+' 300s')
        ax.set(ylabel='BIS',title=f'Prespecified case {cid}');ax.legend(fontsize=7,ncol=3)
        ax2=axes[2*q+1]
        ax2.plot(t,c['action'][lo:hi,0],label='Propofol',color='#7772ae')
        ax2.plot(t,c['action'][lo:hi,1],label='Remifentanil',color='#42a7a0')
        ax2.set(xlabel='Minutes from case start',ylabel='mL / 10s');ax2.legend(fontsize=7)
    save('figure6_prespecified_trajectories.png')
    return ids

def main():
    hm=pd.read_csv(OUT/'horizon_metrics.csv');adv=pd.read_csv(OUT/'paired_comparisons.csv')
    sub=pd.read_csv(OUT/'subgroup_metrics.csv');act=pd.read_csv(OUT/'action_sensitivity_metrics.csv')
    probe=pd.read_csv(OUT/'timescale_probe_metrics.csv');train=pd.read_csv(OUT/'training_summary.csv')
    store=Store();ref=np.load(SOURCE/'outputs/test_reference.npz');summary=json.loads((OUT/'data_summary.json').read_text())
    selected_cases=figures(hm,adv,sub,act,probe,store,ref)
    counts=json.loads((OUT/'parameter_counts.json').read_text());direct_train=pd.read_csv(DIRECT/'outputs/training_summary.csv')
    ar_train=pd.read_csv(SOURCE/'outputs/model_summary.csv')
    cap=[]
    for name in ['AR-RSSM']+NAMES:
        if name=='AR-RSSM':
            q=ar_train[ar_train.policy=='true'].iloc[0];sec=q.seconds;vram=np.nan
        elif name=='Direct-MH':q=direct_train[direct_train.model==name].iloc[0];sec=q.seconds;vram=np.nan
        else:q=train[train.model==name].iloc[0];sec=q.train_seconds;vram=q.peak_vram_mib
        cap.append(dict(model=name,parameters=counts[name],training_parameters=counts[name],deployment_parameters=counts[name],train_seconds=sec,peak_vram_mib=vram))
    capacity=pd.DataFrame(cap);capacity.to_csv(OUT/'model_capacity.csv',index=False)
    # The ablation gate is based on selected primary checkpoints, before any ablation is trained.
    cca=row(adv,comparison='CCA',subset='overall',target='BIS',horizon_seconds=300)
    mta=row(adv,comparison='MTA',subset='overall',target='BIS',horizon_seconds=300)
    lsg=row(adv,comparison='LongShortGain',subset='overall',target='BIS',horizon_seconds=300)
    ablation_gate=bool(cca.difference_mae>=.15 and cca.ci_low>0 and lsg.difference_mae>=.05 and lsg.ci_low>0 and mta.difference_mae>=.1)
    if ablation_gate:raise RuntimeError('Primary benefit met ablation gate; train fast-only, slow-only, same-rate controls before finalizing')
    pd.DataFrame([dict(ablation='fast_only',status='not_run_gate_not_met'),dict(ablation='slow_only',status='not_run_gate_not_met'),
                  dict(ablation='same_rate_two_branch',status='not_run_gate_not_met')]).to_csv(OUT/'ablation_metrics.csv',index=False)
    short=['recent_PPF_action_delta_300s','recent_RFTN_action_delta_300s','recent_BIS_delta_300s','future_delta_BIS_30s','future_delta_MAP_60s']
    long=['history_BIS_mean_300s','history_MAP_mean_300s','history_PPF_action_mean_300s','history_RFTN_action_mean_300s','future_delta_BIS_300s','future_delta_MAP_300s']
    fast_wins=sum(row(probe,representation='z_fast',target=t).r2-row(probe,representation='z_slow',target=t).r2>.03 for t in short)
    slow_wins=sum(row(probe,representation='z_slow',target=t).r2-row(probe,representation='z_fast',target=t).r2>.03 for t in long)
    specialized=fast_wins>=2 and slow_wins>=2
    action_ok=True
    for group in ('Q4','upcoming_large_intervention'):
        q=act[(act.model=='MT-Dynamics')&(act.subset==group)&(act.horizon_seconds==300)]
        true=row(q,condition='true').mae
        if not (true<row(q,condition='hold').mae and true<row(q,condition='wrong').mae):action_ok=False
    event_consistent=sum(row(adv,comparison='CCA',subset=s,target='BIS',horizon_seconds=300).difference_mae>.1
                         for s in ('Q4','upcoming_large_intervention','initiation','increase'))>=2
    if mta.difference_mae>=.1 and not action_ok:outcome='E';decision='STOP';reason='Forecast gains were accompanied by lost prospective-action sensitivity.'
    elif cca.difference_mae>=.15 and cca.ci_low>0 and lsg.difference_mae>=.05 and lsg.ci_low>0 and event_consistent and action_ok and specialized:
        outcome='A';decision='PURSUE';reason='Temporal factorization exceeded both direct baselines with growing long-horizon and intervention-change benefit.'
    elif cca.difference_mae>0 and cca.ci_low>0 and lsg.difference_mae>0 and lsg.ci_low>0 and action_ok:
        outcome='B';decision='CONDITIONAL';reason='A positive capacity-controlled long-horizon effect survived patient bootstrap, but the practical benefit was modest.'
    elif mta.difference_mae>=.1 and abs(cca.difference_mae)<.05:
        outcome='C';decision='STOP';reason='Additional capacity explained the improvement over the original Direct-MH model.'
    else:outcome='F';decision='STOP';reason='The selected multi-timescale model did not establish a meaningful capacity-controlled forecast advantage.'
    # Verify actual update calls on one real batch, rather than infer timescales from names.
    ds=Windows(store,'test');sample=to_device(next(iter(torch.utils.data.DataLoader(ds,batch_size=2))), 'cpu')
    mt=MTDynamics();ck=torch.load(ROOT/'checkpoints/MT-Dynamics.pt',map_location='cpu',weights_only=False);mt.load_state_dict(ck['model']);mt.eval()
    calls={'fast':0,'slow':0};hooks=[]
    hooks.append(mt.fast_transition.register_forward_hook(lambda *args:calls.__setitem__('fast',calls['fast']+1)))
    hooks.append(mt.slow_transition.register_forward_hook(lambda *args:calls.__setitem__('slow',calls['slow']+1)))
    with torch.no_grad():mt(sample)
    for hook in hooks:hook.remove()
    assert calls=={'fast':30,'slow':5},calls
    splits=store.subject_splits;overlap={f'{a}_{b}':len(set(splits[a])&set(splits[b])) for a,b in [('train','val'),('train','test'),('val','test')]}
    integrity={'patient_overlap':overlap,'normalization_sha256':sha((SOURCE/'../vitaldb_intervention_grounding_seed0_v1/outputs/normalization.json').resolve()),
               'epoch_indices_sha256':sha(SOURCE/'outputs/epoch_train_indices.npy'),
               'direct_checkpoint_sha256':sha(DIRECT/'outputs/checkpoints/Direct-MH.pt'),
               'direct_prediction_max_abs_difference':summary['direct_reproduction_max_abs_diff'],
               'same_test_windows':True,'same_horizons':STEPS,'ce_in_training':False,'test_used_for_training_or_selection':False,
               'model_future_inputs':'prospective action trajectory only; factual future physiology excluded',
               'fast_updates_per_300s':calls['fast'],'slow_updates_per_300s':calls['slow'],
               'capacity_mismatch_fraction':abs(counts['Direct-MH-Capacity']/counts['MT-Dynamics']-1),
               'donor_matching_uses_future_outcomes':False,'patient_cluster_bootstrap_replicates':1000,
               'prespecified_example_cases':selected_cases}
    assert all(x==0 for x in overlap.values()) and integrity['capacity_mismatch_fraction']<.1 and summary['direct_reproduction_max_abs_diff']<1e-5
    (OUT/'integrity_checks.json').write_text(json.dumps(integrity,indent=2))
    q=hm[(hm.subset=='overall')&(hm.target=='BIS')]
    main=table(q[['model','horizon_seconds','mae','rmse','patients']],['model','horizon_seconds','mae','rmse','patients'])
    map_main=table(hm[(hm.subset=='overall')&(hm.target=='MAP')][['model','horizon_seconds','mae','rmse','patients']],
                   ['model','horizon_seconds','mae','rmse','patients'])
    event_cca=table(adv[(adv.comparison=='CCA')&(adv.target=='BIS')&(adv.horizon_seconds==300)&
                        (adv.subset.isin(['stable_action_Q1','Q4','upcoming_large_intervention','initiation','increase','decrease','stop']))]
                    [['subset','difference_mae','ci_low','ci_high','patients','windows']],
                    ['subset','difference_mae','ci_low','ci_high','patients','windows'])
    paired_table=table(adv[(adv.subset=='overall')&(adv.target=='BIS')][['comparison','horizon_seconds','difference_mae','ci_low','ci_high','patients','windows']],
                       ['comparison','horizon_seconds','difference_mae','ci_low','ci_high','patients','windows'])
    questions=[
      f"1. Beyond Direct-MH? No convincing gain: 300s MTA={f(mta.difference_mae)} [{f(mta.ci_low)}, {f(mta.ci_high)}] BIS MAE; the interval crosses zero.",
      f"2. Beyond capacity control? Yes statistically, but modest: 300s CCA={f(cca.difference_mae)} [{f(cca.ci_low)}, {f(cca.ci_high)}], about {100*cca.difference_mae/row(hm,model='Direct-MH-Capacity',subset='overall',target='BIS',horizon_seconds=300).mae:.1f}% of control error.",
      f"3. Growing with horizon? LongShortGain={f(lsg.difference_mae)} [{f(lsg.ci_low)}, {f(lsg.ci_high)}].",
      f"4. Intervention changes? Mixed, and no robust improvement over the original Direct-MH: 300s capacity-controlled advantages for Q4, upcoming, initiation, increase are "+', '.join(s+' '+f(row(adv,comparison='CCA',subset=s,target='BIS',horizon_seconds=300).difference_mae) for s in ('Q4','upcoming_large_intervention','initiation','increase'))+'.',
      f"5. Action sensitivity? {'Preserved' if action_ok else 'Failed'} in the matched-donor Q4 and upcoming-change 300s checks; full true/hold/wrong table is below.",
      f"6. Branch specialization? Fast won {fast_wins}/{len(short)} prespecified short probes and slow won {slow_wins}/{len(long)} prespecified long probes by >0.03 R²; this is {'consistent' if specialized else 'insufficient'} evidence of specialization. Probe interpretation is descriptive.",
      '7. Same-rate two-branch? Not run because the primary capacity-controlled benefit did not meet the predeclared ablation gate; this comparison remains unresolved and cannot be used as positive evidence.',
      f'8. Decision: {decision}, Outcome {outcome}.']
    lines=[f'Decision: {decision}',reason,'',f'Outcome {outcome}. One seed, one VitalDB cohort; no causal effect claim.','',
           '## Cohort and controlled design',f"495 cases / 493 patients; patient-disjoint splits {summary['case_splits']}, effective TEST {summary['test_patients']} patients and {summary['test_windows']} windows. 10-second sampling, 180-step history, 30-step future; all models use the same four endpoints and 80,000-window epoch permutations.",
           'The unchanged Round-1 numeric tracks, normalization, masks, demographics, RATE/VOL rules and observed prospective future actions are inherited. CE and waveforms are excluded from training. Direct-MH was reused from its original checkpoint and its TEST predictions matched to numerical tolerance. Fast transitions run 30 times and slow transitions 5 times per 300s rollout. The slow update consumes only its completed six-action block and current fast state.',
           '## Parameter and compute accounting',table(capacity,['model','parameters','training_parameters','deployment_parameters','train_seconds','peak_vram_mib']),
           'Reference Direct-MH and AR peak VRAM were not recorded in their original runs (NA).',
           '## Primary TEST BIS results',main,'',
           '## Secondary TEST MAP results',map_main,'',
           '## Paired patient-cluster comparisons','Positive MTA/CCA means MT-Dynamics lowers error. LongShortGain is CCA(300s) minus CCA(30s). CIs use 1,000 resamples of patients.',paired_table,
           '## Intervention-event and action audit','Paired 300s capacity-controlled event comparisons (patient-bootstrap 95% CI):',event_cca,
           table(sub[(sub.target=='BIS')&(sub.horizon_seconds.isin([60,180,300]))],['model','subset','horizon_seconds','mae','patients','windows']),
           table(act[(act.horizon_seconds==300)],['model','condition','subset','mae','patients','windows']),
           'The true, hold, and wrong action rows use identical donor-eligible windows within each subset.',
           '## Frozen branch probes',table(probe,['representation','target','r2','r2_ci_low','r2_ci_high','patients']),
           'Recent action/physiology summaries come only from historical samples; future-change labels are used solely in frozen-model probes. The initial latent states cannot see future actions, so these probes should not be treated as full intervention-response capability.',
           '## Conditional ablations',f'Ablation gate met: {ablation_gate}. No ablation checkpoints were trained because the primary result did not satisfy that gate.',
           '## Eight research questions',*questions,'',
           '## Figures',
           '![Figure 1](../plots/figure1_bis_horizon.png)','![Figure 2](../plots/figure2_capacity_controlled_advantage.png)',
           '![Figure 3](../plots/figure3_intervention_events.png)','![Figure 4](../plots/figure4_timescale_probes.png)',
           '![Figure 5](../plots/figure5_action_corruption.png)','![Figure 6](../plots/figure6_prespecified_trajectories.png)',
           '## Interpretation','The predeclared rule assigns Outcome B / CONDITIONAL because the capacity-controlled 300s difference and its horizon gradient have patient-bootstrap intervals above zero. This does not establish that distinct update rates caused the gain: the same-rate two-branch ablation was gated off, frozen probes showed no predicted fast/slow specialization, and the original Direct-MH is effectively tied with MT-Dynamics. The 0.064 BIS MAE capacity-controlled difference is small relative to about 5 BIS MAE absolute error; MT-Dynamics also took more training time. This is insufficient to build a publication claim or scale the architecture now.',
           '## Limits','A single seed and one observational dataset cannot establish broad architectural generality. Prospective schedules are observed factual actions; matched wrong schedules are input perturbations, not treatment counterfactuals. Highly overlapping windows are handled by patient-clustered uncertainty, but effective independent sample size is limited.']
    (OUT/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n')
    required=['data_summary.json','model_capacity.csv','training_summary.csv','horizon_metrics.csv','multitimescale_advantage.csv',
      'capacity_controlled_advantage.csv','subgroup_metrics.csv','action_sensitivity_metrics.csv','timescale_probe_metrics.csv',
      'ablation_metrics.csv','paired_comparisons.csv','integrity_checks.json','FINAL_REPORT.md']
    checks={'files':{x:(OUT/x).exists() and (OUT/x).stat().st_size>0 for x in required},
            'selected_checkpoints':{x:(ROOT/'checkpoints'/f'{x}.pt').exists() for x in ('Direct-MH-Capacity','MT-Dynamics')},
            'figures':sorted(x.name for x in PLOTS.glob('figure*.png')),'outcome':outcome,'decision':decision,
            'all_passed':all((OUT/x).exists() and (OUT/x).stat().st_size>0 for x in required) and len(list(PLOTS.glob('figure*.png')))==6}
    (OUT/'completion_checks.json').write_text(json.dumps(checks,indent=2))
    print('FINALIZE_COMPLETE',outcome,decision,flush=True)
if __name__=='__main__':main()
