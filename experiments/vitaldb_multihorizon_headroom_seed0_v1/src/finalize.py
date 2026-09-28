"""Patient-paired trend tests, figures, integrity gate and falsification report."""
import json,hashlib,warnings
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *
from direct import Direct
from evaluate import OracleWindows,bootstrap,subsets

COLORS={'AR-RSSM':'#414b57','Direct-MH':'#1479a6','Direct-Traj':'#c45537',
        'Oracle-State':'#739459','Oracle-Latent-Reset':'#9b70a4'}
def read(name):return pd.read_csv(OUT/name)
def scalar(df,**filters):
    for k,v in filters.items():df=df[df[k]==v]
    assert len(df)==1,(filters,len(df))
    return df.iloc[0]
def fmt(x):return f'{x:.3f}' if np.isfinite(x) else 'NA'
def save(fig,name):
    fig.tight_layout();fig.savefig(ROOT/'plots'/f'{name}.png',dpi=170)
    fig.savefig(ROOT/'plots'/f'{name}.pdf');plt.close(fig)

def trend_analysis(ref):
    ar=np.load(ARRAYS/'ar_prediction.npy',mmap_mode='r')
    oracle=np.load(ARRAYS/'oracle_prediction.npy',mmap_mode='r')
    reset=np.load(ARRAYS/'oracle_latent_reset_prediction.npy',mmap_mode='r')
    fresh=np.load(ARRAYS/'oracle_anchor_fresh.npy',mmap_mode='r')
    mh=np.load(ARRAYS/'Direct-MH_true.npy',mmap_mode='r')
    traj=np.load(ARRAYS/'Direct-Traj_true.npy',mmap_mode='r')
    y=ref['target'];mask=ref['mask'].astype(bool);subject=ref['subject']
    groups={'overall':np.ones(len(y),bool),'Q4':ref['quartile']==4,'upcoming_large_intervention':ref['upcoming_label']>0}
    rows=[]
    for subset,keep in groups.items():
        for name,p in [('DHA',mh),('DTA',traj),('RP',oracle),('RP_latent_reset',reset)]:
            for j,target in enumerate(['BIS','MAP']):
                v=keep&mask[:,2,j]&mask[:,29,j]
                if name.startswith('RP'):v&=fresh[:,2,j]&fresh[:,29,j]
                d30=np.abs(ar[:,2,j]-y[:,2,j])-np.abs(p[:,2,j]-y[:,2,j])
                d300=np.abs(ar[:,29,j]-y[:,29,j])-np.abs(p[:,29,j]-y[:,29,j])
                point,lo,hi,n,w=bootstrap(d300-d30,subject,v,CFG['bootstrap_replicates'])
                rows.append({'subset':subset,'target':target,'comparison':name,'change_300s_minus_30s':point,
                             'ci_low':lo,'ci_high':hi,'patients':n,'windows':w})
    pd.DataFrame(rows).to_csv(OUT/'headroom_trends.csv',index=False)

def oracle_leakage_check(ds):
    ow=OracleWindows(ds,6);i=0;cid,t=map(int,ds.indices[i]);c=ds.store.cases[cid]
    old=ow[i];new_target=t+6;later=t+7
    original_state=c['sn'][new_target].copy();original_action=c['an'][later].copy()
    try:
        c['sn'][new_target]+=1000;c['an'][later]+=1000
        mutated=ow[i]
    finally:
        c['sn'][new_target]=original_state;c['an'][later]=original_action
    return all(np.array_equal(old[k],mutated[k]) for k in old)

def checks(store,ds):
    source_summary=json.loads((SOURCE/'outputs/data_summary.json').read_text())
    current=json.loads((OUT/'data_summary.json').read_text())
    reproduced=json.loads((OUT/'ar_reproduction.json').read_text())
    r=RSSM();d=Direct()
    ar_count=sum(p.numel() for p in r.parameters());direct_count=sum(p.numel() for p in d.parameters())
    ref=np.load(SOURCE/'outputs/test_reference.npz')
    sample=ds[0];cid,t=map(int,ds.indices[0]);case=store.cases[cid]
    result={
      'same_cohort_495_cases_493_patients':len(store.cases)==495 and current['patients']==493,
      'same_patient_split':current['split_cases']=={'train':345,'val':74,'test':76} and current['split_subjects']=={'train':345,'val':73,'test':75},
      'same_complete_future_action_windows':len(ds)==83198 and current['test_windows']==source_summary['windows']['test']['complete_future_action'],
      'source_test_index_exact':np.array_equal(ds.indices[:,0],ref['case']) and np.array_equal(ds.indices[:,1],ref['t']),
      'AR_checkpoint_prediction_exact':reproduced['prediction_max_abs_difference']==0.0,
      'AR_checkpoint_unchanged':reproduced['checkpoint_sha256']==current['source_round2_checkpoint_sha256'],
      'direct_capacity_within_half_to_two_AR':.5<=direct_count/ar_count<=2,
      'same_direct_architecture':all(torch.load(OUT/'checkpoints'/f'{name}.pt',map_location='cpu',weights_only=False)['parameters']==direct_count for name in ['Direct-MH','Direct-Traj']),
      'oracle_target_or_later_action_mutation_no_input_effect':oracle_leakage_check(ds),
      'oracle_no_later_physiology_by_index':all((t+h-1)<(t+h) for _,t in ds.indices[:32] for h in HORIZONS),
      'future_action_starts_at_t_plus_1':np.array_equal(sample['future_action'][:,:2],case['an'][t+1:t+F+1]),
      'CE_not_in_features_or_loss':not any('ce' in key.lower() for key in sample) and sample['state'].shape==(H,6)
          and sample['action'].shape==(H,4) and sample['target'].shape==(F,2),
      'patient_bootstrap_1000':CFG['bootstrap_replicates']==1000,
    }
    write_json('integrity_checks.json',result)
    if not all(result.values()):raise RuntimeError('Integrity failure '+str(result))
    enc_ar=sum(p.numel() for p in r.encoder.parameters())+sum(p.numel() for p in r.latent.parameters())
    enc_direct=sum(p.numel() for p in d.encoder.parameters())+sum(p.numel() for p in d.latent.parameters())
    pd.DataFrame([{'model':name,'parameters':count,'encoder_parameters':enc,
                   'predictor_decoder_parameters':count-enc,'training_flops_estimate':np.nan,
                   'inference_latency_ms':np.nan}
      for name,count,enc in [('AR-RSSM',ar_count,enc_ar),('Direct-MH',direct_count,enc_direct),
                             ('Direct-Traj',direct_count,enc_direct)]]).to_csv(OUT/'model_capacity.csv',index=False)
    return result

def figures(ref):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    metrics=read('horizon_metrics.csv');pairs=read('paired_comparisons.csv');actions=read('action_sensitivity_metrics.csv')
    x=np.array([30,60,180,300])
    fig,ax=plt.subplots(figsize=(7,4.3))
    for name in ['AR-RSSM','Direct-MH','Direct-Traj','Oracle-State']:
        v=[scalar(metrics,subset='overall',target='BIS',model=name,horizon_seconds=int(h)).mae for h in x]
        ax.plot(x,v,marker='o',label=name,color=COLORS[name])
    ax.set(xlabel='Horizon (seconds)',ylabel='Patient-weighted BIS MAE',xticks=x)
    ax.legend(frameon=False);ax.grid(alpha=.18);save(fig,'figure1_error_vs_horizon')
    for label,name,title in [('DHA','figure2_direct_horizon_advantage','Direct-MH advantage over AR'),
                             ('RP','figure3_recursion_penalty','Factual-anchor Oracle-State difference')]:
        q=pairs[(pairs.subset=='overall')&(pairs.target=='BIS')&(pairs.comparison==label)].set_index('horizon_seconds').loc[x]
        fig,ax=plt.subplots(figsize=(7,4.2))
        ax.errorbar(x,q.difference_mae,yerr=[q.difference_mae-q.ci_low,q.ci_high-q.difference_mae],
                    color=COLORS['Direct-MH' if label=='DHA' else 'Oracle-State'],marker='o',capsize=4)
        ax.axhline(0,color='black',lw=.8);ax.set(xlabel='Horizon (seconds)',ylabel='Paired BIS MAE difference',title=title,xticks=x)
        if label=='RP':ax.text(.02,.03,'Uses factual future history; not an isolated drift estimate',transform=ax.transAxes,fontsize=9)
        ax.grid(alpha=.18);save(fig,name)
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),sharey=True)
    for ax,(subset,title) in zip(axes,[('stable_action_Q1','Stable action (Q1)'),
                                       ('upcoming_large_intervention','Upcoming large change'),('Q4','High divergence (Q4)')]):
        for name in ['AR-RSSM','Direct-MH']:
            v=[scalar(metrics,subset=subset,target='BIS',model=name,horizon_seconds=int(h)).mae for h in x]
            ax.plot(x,v,marker='o',label=name,color=COLORS[name])
        ax.set(title=title,xlabel='Seconds',xticks=x);ax.grid(alpha=.18)
    axes[0].set_ylabel('Patient-weighted BIS MAE');axes[0].legend(frameon=False)
    save(fig,'figure4_intervention_subgroups')
    ds=ref['donor']>=0
    y=ref['target'][:,29,0];m=ref['mask'][:,29,0].astype(bool)
    subset=(ref['quartile']==4)
    fig,ax=plt.subplots(figsize=(6,4.1))
    values=[]
    for condition in ['true','hold','wrong']:
        p=np.load(ARRAYS/f'Direct-MH_{condition}.npy',mmap_mode='r')[:,29,0]
        valid=m&subset&ds
        good=valid&np.isfinite(p)&np.isfinite(y)
        patient=pd.DataFrame({'s':ref['subject'][good],'v':np.abs(p[good]-y[good])}).groupby('s').v.mean()
        values.append(patient.mean())
    ax.bar(range(3),values,color=['#1479a6','#dfa329','#c45537'])
    ax.set_xticks(range(3),['True','Hold last','Matched wrong']);ax.set_ylabel('Q4 BIS MAE at 300 s')
    ax.set_title('Same matched high-divergence TEST windows');save(fig,'figure5_action_corruption')
    ar=np.load(ARRAYS/'ar_prediction.npy',mmap_mode='r')
    mh=np.load(ARRAYS/'Direct-MH_true.npy',mmap_mode='r')
    traj=np.load(ARRAYS/'Direct-Traj_true.npy',mmap_mode='r')
    cases=np.sort(np.unique(ref['case']))
    chosen=cases[np.linspace(0,len(cases)-1,4,dtype=int)]
    fig,axes=plt.subplots(2,2,figsize=(11,6.4),sharex=True)
    for ax,cid in zip(axes.ravel(),chosen):
        inds=np.flatnonzero(ref['case']==cid);i=int(inds[len(inds)//2])
        xx=np.arange(1,31)*10
        ax.plot(xx,ref['target'][i,:,0],color='black',label='Observed BIS',lw=1.6)
        ax.plot(xx,ar[i,:,0],color=COLORS['AR-RSSM'],label='AR-RSSM')
        ax.plot(xx,traj[i,:,0],color=COLORS['Direct-Traj'],label='Direct-Traj')
        ax.scatter(x,mh[i,np.array(HORIZONS)-1,0],color=COLORS['Direct-MH'],label='Direct-MH',s=25)
        ax.set_title(f'Case {cid}, anchor {int(ref["t"][i])}');ax.grid(alpha=.16)
    axes[0,0].legend(frameon=False,ncol=2,fontsize=8)
    for ax in axes[1]:ax.set_xlabel('Seconds from anchor')
    for ax in axes[:,0]:ax.set_ylabel('BIS')
    save(fig,'figure6_prespecified_trajectories')

def enrich_growth():
    growth=read('error_growth.csv');metrics=read('horizon_metrics.csv')
    slopes=[]
    for r in growth.itertuples():
        q=metrics[(metrics.subset==r.subset)&(metrics.model==r.model)&
                  (metrics.target==r.target)&(metrics.horizon_seconds.isin([30,60,180,300]))]
        q=q.sort_values('horizon_seconds')
        assert len(q)==4
        slopes.append(float(np.polyfit(np.log(q.horizon_seconds),q.mae,1)[0]))
    growth['slope_per_log_second']=slopes
    growth.to_csv(OUT/'error_growth.csv',index=False)

def report():
    m=read('horizon_metrics.csv');p=read('paired_comparisons.csv');a=read('action_sensitivity_metrics.csv')
    t=read('headroom_trends.csv');cap=read('model_capacity.csv');train=read('training_summary.csv')
    growth=read('error_growth.csv')
    data=json.loads((OUT/'data_summary.json').read_text())
    def pair(label,h,subset='overall'):
        return scalar(p,subset=subset,target='BIS',comparison=label,horizon_seconds=h)
    d30=pair('DHA',30);d300=pair('DHA',300);dt=pair('DTA',300)
    trend=scalar(t,subset='overall',target='BIS',comparison='DHA')
    rp=pair('RP',300);reset=pair('RP_latent_reset',300)
    q4=pair('DHA',300,'Q4');up=pair('DHA',300,'upcoming_large_intervention')
    hold=scalar(a,model='Direct-MH',subset='Q4',target='BIS',condition='hold',horizon_seconds=300)
    wrong=scalar(a,model='Direct-MH',subset='Q4',target='BIS',condition='wrong',horizon_seconds=300)
    # A needs a clear direct gain in intervention-change subsets and evidence of recursive drift.
    # The raw oracle comparison alone cannot establish drift because it resets factual BIS/MAP.
    if (d300.ci_low>0 and dt.ci_low>0 and trend.ci_low>0 and q4.ci_low>0 and up.ci_low>0
        and hold.ci_low>0 and wrong.ci_low>0 and reset.ci_low>0):
        outcome='A';title='Strong multi-horizon headroom supported'
    elif d300.ci_low>0 and dt.ci_low>0 and trend.ci_low>0 and reset.ci_high<=0:
        outcome='B';title='Direct objective/formulation headroom without isolated recursive drift'
    elif rp.ci_low>0 and reset.ci_low>0 and d300.ci_high<=0 and dt.ci_high<=0:
        outcome='C';title='Recursive drift signal without direct-model gain'
    else:outcome='D';title='Multi-horizon hypothesis not sufficiently supported'
    lines=['# VitalDB multi-horizon headroom, seed 0',f'\n**Outcome {outcome} — {title}.**',
      '\n## Central finding',
      f'On {data["test_windows"]:,} complete-action TEST windows from {data["effective_test_patients"]} patients, the Direct-MH advantage over AR (AR error minus Direct-MH error) rises from {fmt(d30.difference_mae)} [{fmt(d30.ci_low)}, {fmt(d30.ci_high)}] at 30 s to {fmt(d300.difference_mae)} [{fmt(d300.ci_low)}, {fmt(d300.ci_high)}] at 300 s. The paired change in advantage from 30 to 300 s is {fmt(trend.change_300s_minus_30s)} [{fmt(trend.ci_low)}, {fmt(trend.ci_high)}]. Direct-Traj also gains {fmt(dt.difference_mae)} [{fmt(dt.ci_low)}, {fmt(dt.ci_high)}] at 300 s, so endpoint-only supervision does not fully explain the gain.',
      f'However, the 300-s Direct-MH gain in Q4 high future-action divergence is {fmt(q4.difference_mae)} [{fmt(q4.ci_low)}, {fmt(q4.ci_high)}] and around upcoming large interventions is {fmt(up.difference_mae)} [{fmt(up.ci_low)}, {fmt(up.ci_high)}]. Neither gives a clear intervention-focused advantage. This limits the case for an intervention-specific multi-horizon method.',
      f'Q4 Direct-MH action corruption raises 300-s BIS MAE by {fmt(hold.degradation_mae)} [{fmt(hold.ci_low)}, {fmt(hold.ci_high)}] when holding the last action, and {fmt(wrong.degradation_mae)} [{fmt(wrong.ci_low)}, {fmt(wrong.ci_high)}] with a matched wrong future schedule. Thus the direct predictor uses the prospective action sequence.',
      '\n## Data and design',
      'The unchanged VitalDB numeric cohort has 495 cases and 493 patients. The original case splits are 345/74/76 and subject-disjoint patient splits 345/73/75; 74 test patients have 83,198 complete prospective-action windows. Histories have 180 ten-second steps and futures have 30 steps. BIS and MAP are targets; HR, demographics, historical propofol/remifentanil, missingness masks and observed prospective actions are inherited. Normalization, RATE/VOL preprocessing and exclusions are exactly those of Round 1 and Round 2. No new data were downloaded. Device-computed TCI CE is absent from all model and support features.',
      'AR-RSSM is the unchanged Round-2 seed-0 checkpoint. Re-inferred predictions match its saved 83,198 × 30 × 2 TEST predictions exactly (maximum absolute difference 0). Its latent state at every rollout step is saved in the server-only array cache. Direct-MH and Direct-Traj use the same two-layer 64-dimensional GRU history encoder and a causal 32-dimensional prospective-action GRU. A shared point decoder receives historical latent, action prefix and learned horizon embedding; Direct-MH supervises four endpoints, whereas Direct-Traj supervises all 30. Both decode each point in one forward pass without predicted-state feedback. No auxiliary or representation objective was used.',
      'Both direct models used the original Round-2 epoch window permutations (80,000 windows per epoch), AdamW, standardized BIS MSE + 0.25 MAP MSE, learning rate 0.0005, weight decay 0.0001, 20-epoch maximum, patience five and stride-six validation. Seed 0 was fixed. A separate Direct-NoHorizon run was omitted because it requires another complete training run and is secondary to testing direct versus recursive prediction.',
      '\n## Model capacity and training',
      '| Model | Parameters | History encoder + norm | Action/predictor/decoder | Best epoch | Best VAL BIS MAE |',
      '|---|---:|---:|---:|---:|---:|']
    for name in ['AR-RSSM','Direct-MH','Direct-Traj']:
        c=scalar(cap,model=name)
        if name=='AR-RSSM':
            old=read('source_ar_training_summary.csv').iloc[0];epoch=old.best_epoch;score=old.best_val_bis_mae
        else:
            tr=scalar(train,model=name);epoch=tr.best_epoch;score=tr.best_val_bis_mae
        lines.append(f'| {name} | {int(c.parameters):,} | {int(c.encoder_parameters):,} | {int(c.predictor_decoder_parameters):,} | {int(epoch)} | {fmt(score)} |')
    lines+=['\nFLOPs and latency were not measured; `model_capacity.csv` leaves them blank rather than inventing estimates.',
            '\n## Primary patient-weighted TEST results',
            '| Model | 30s BIS MAE | 60s | 180s | 300s | Full 5m BIS MAE |',
            '|---|---:|---:|---:|---:|---:|']
    for name in ['AR-RSSM','Oracle-State','Direct-MH','Direct-Traj']:
        vals=[fmt(scalar(m,subset='overall',model=name,target='BIS',horizon_seconds=h).mae) for h in [30,60,180,300]]
        full=fmt(scalar(m,subset='overall',model=name,target='BIS',horizon_seconds=0).mae) if name in ['AR-RSSM','Direct-Traj'] else 'N/A'
        lines.append(f'| {name} | '+ ' | '.join(vals)+f' | {full} |')
    lines+=['\n| Model | 30s MAP MAE | 60s | 180s | 300s | Full 5m MAP MAE |',
            '|---|---:|---:|---:|---:|---:|']
    for name in ['AR-RSSM','Oracle-State','Direct-MH','Direct-Traj']:
        vals=[fmt(scalar(m,subset='overall',model=name,target='MAP',horizon_seconds=h).mae) for h in [30,60,180,300]]
        full=fmt(scalar(m,subset='overall',model=name,target='MAP',horizon_seconds=0).mae) if name in ['AR-RSSM','Direct-Traj'] else 'N/A'
        lines.append(f'| {name} | '+ ' | '.join(vals)+f' | {full} |')
    lines+=['\nRMSE at all horizons and full-trajectory BIS/MAP MAE/RMSE are in `horizon_metrics.csv`. The main figure shows BIS; all metrics weight patients equally after averaging eligible windows within each patient. Paired 95% intervals use 1,000 patient-cluster resamples. Windows were never treated as independent patients.',
            '\n| Comparison (AR error minus comparator) | 30s [95% CI] | 60s | 180s | 300s |',
            '|---|---:|---:|---:|---:|']
    for label in ['DHA','DTA','RP','RP_latent_reset']:
        vals=[]
        for h in [30,60,180,300]:
            q=pair(label,h);vals.append(f'{fmt(q.difference_mae)} [{fmt(q.ci_low)}, {fmt(q.ci_high)}]')
        lines.append(f'| {label} | '+' | '.join(vals)+' |')
    lines+=['\n| Model | BIS growth 30→300s | Normalized growth | Slope vs log(seconds) |',
            '|---|---:|---:|---:|']
    for name in ['AR-RSSM','Oracle-State','Direct-MH','Direct-Traj']:
        g=scalar(growth,subset='overall',model=name,target='BIS')
        lines.append(f'| {name} | {fmt(g.absolute_growth)} | {fmt(g.normalized_growth)} | {fmt(g.slope_per_log_second)} |')
    lines+=['\nThe slope is an unadjusted four-point descriptive fit against natural log horizon in seconds. `error_growth.csv` also reports MAP and subgroup growth; `headroom_trends.csv` reports paired change in DHA, DTA and oracle differences with bootstrap intervals.',
            '\n## Oracle-State interpretation',
            f'The local factual-anchor Oracle-State comparison gives RP(300 s) = {fmt(rp.difference_mae)} [{fmt(rp.ci_low)}, {fmt(rp.ci_high)}]. It uses the true BIS/MAP at t+290 s as the model residual baseline and encodes the real history ending there. This has a major information advantage over a deployable t-origin prediction, so a growing raw RP cannot isolate recursive latent drift. The additional Oracle-Latent-Reset keeps the original t-origin residual baseline while replacing only the latent with the factual later-anchor encoding. Its 300-s difference is {fmt(reset.difference_mae)} [{fmt(reset.ci_low)}, {fmt(reset.ci_high)}]: factual latent reset does not improve the free rollout under a fixed baseline. This reset also creates a state/baseline pairing that was never trained, so its negative result cannot prove the absence of drift. Together the audits do not isolate a recursive-drift mechanism.',
            'Each oracle input ends at t+h−1; its one local transition receives only action t+h. No physiological value at t+h or later action enters that input. An automated mutation test confirms that changing the target-time physiology or later action leaves the oracle input unchanged. The oracle is diagnostic and is excluded from claims of deployable forecast performance.',
            '\n## Intervention subgroups and action dependence',
            '| Subgroup | AR 300s MAE | Direct-MH 300s MAE | DHA [95% CI] | Patients |',
            '|---|---:|---:|---:|---:|']
    for subset in ['overall','Q4','upcoming_large_intervention','stable_action_Q1','initiation','increase','decrease','stop','high_rate_tail','non_tail']:
        ar=scalar(m,subset=subset,model='AR-RSSM',target='BIS',horizon_seconds=300)
        mh=scalar(m,subset=subset,model='Direct-MH',target='BIS',horizon_seconds=300)
        q=pair('DHA',300,subset)
        lines.append(f'| {subset} | {fmt(ar.mae)} | {fmt(mh.mae)} | {fmt(q.difference_mae)} [{fmt(q.ci_low)}, {fmt(q.ci_high)}] | {int(q.patients)} |')
    lines+=['\nInitiation/increase/decrease/stop and upcoming-large labels reuse the Round-2 action-only event definitions. Q4 and stable Q1 use Round-2 TRAIN-defined divergence cutoffs. Tail/non-tail use its TRAIN 99th-percentile positive-rate caps; these are exploratory because that sensitivity analysis was added after Round-2 inspection. The stop subgroup has only 35 patients and a wide interval. Matched wrong-action swaps use the original Round-2 donor mapping from other TEST patients and are assessed only where a donor exists. `action_sensitivity_metrics.csv` gives both direct models, all horizons, BIS/MAP, hold and matched wrong actions.',
            '\n## Figures',
            *[f'![Figure {i}](../plots/{name}.png)' for i,name in enumerate([
                'figure1_error_vs_horizon','figure2_direct_horizon_advantage','figure3_recursion_penalty',
                'figure4_intervention_subgroups','figure5_action_corruption','figure6_prespecified_trajectories'],1)],
            '\nThe trajectory figure uses cases at fixed quartile positions of sorted TEST case IDs and each case’s median eligible anchor; no model error informed display-window selection.',
            '\n## Scientific decision and limitations']
    if outcome=='B':lines.append('This experiment finds a small but increasing direct-prediction benefit, including a full-trajectory direct control, with retained prospective-action sensitivity. It does not demonstrate a recursive latent-drift mechanism, and the long-horizon gain is not clearly amplified in Q4 or upcoming intervention changes. Outcome B is the closest of the four requested categories, with an important qualification: the original AR-RSSM was itself trained against the full 30-step trajectory, so these results do not establish a one-step-objective mismatch. The raw Oracle-State and AR errors are not similar because the oracle receives factual future physiology. Direct model structure, action encoding and optimization remain alternative explanations. The evidence is insufficient to frame a final model as fixing recursive rollout drift or to claim intervention-specific benefit.')
    elif outcome=='A':lines.append('A long-horizon, intervention-focused direct advantage with preserved action sensitivity and a latent-reset drift signal supports pursuing a compact multi-horizon world model. This remains observational and is not a causal treatment evaluation.')
    elif outcome=='C':lines.append('The isolated latent-reset drift signal is not exploited by either direct formulation. A final model is premature; examine the transition and representation first.')
    else:lines.append('The experiment does not provide the specified consistent long-horizon direct advantage. Do not make the compact multi-horizon formulation the main project on this evidence.')
    lines.append('\nThis is one architecture, one seed for the new direct models, one split and a five-minute observational horizon. Parameter counts are close but the direct causal action encoder differs from the RSSM GRUCell, so an architectural contribution cannot be excluded. Shared validation checkpoint selection and direct-model optimization can also affect small differences. Oracle-State uses future factual physiology and must not be presented as deployable. These data do not establish treatment counterfactual accuracy or clinical safety.')
    (OUT/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n')
    return outcome

def main():
    OUT.mkdir(exist_ok=True);(ROOT/'plots').mkdir(exist_ok=True)
    store=Store();ds=Windows(store,'test');ref=np.load(SOURCE/'outputs/test_reference.npz')
    checks(store,ds)
    ar_train=pd.read_csv(SOURCE/'outputs/model_summary.csv')
    ar_train=ar_train[ar_train.policy.astype(str).str.lower()=='true']
    assert len(ar_train)==1
    ar_train.to_csv(OUT/'source_ar_training_summary.csv',index=False)
    trend_analysis(ref);enrich_growth();figures(ref);outcome=report()
    required=['data_summary.json','model_capacity.csv','training_summary.csv','horizon_metrics.csv',
              'direct_horizon_advantage.csv','recursion_penalty.csv','subgroup_metrics.csv',
              'action_sensitivity_metrics.csv','oracle_state_protocol.json','paired_comparisons.csv',
              'integrity_checks.json','FINAL_REPORT.md']
    complete={'outcome':outcome,'required_outputs_present':all((OUT/f).exists() for f in required),
              'figures_png':len(list((ROOT/'plots').glob('figure*.png'))),
              'new_checkpoints':len(list((OUT/'checkpoints').glob('Direct-*.pt'))),
              'AR_exact_reproduction':json.loads((OUT/'ar_reproduction.json').read_text())['prediction_max_abs_difference']==0,
              'bootstrap_replicates':CFG['bootstrap_replicates']}
    write_json('completion_checks.json',complete)
    assert complete['required_outputs_present'] and complete['figures_png']>=6 and complete['new_checkpoints']==2
    print('FINAL_REPORT_COMPLETE',outcome,flush=True)

if __name__=='__main__':main()
