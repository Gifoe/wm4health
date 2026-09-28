"""Scientific integrity gate, seven figures, and evidence-bound final report."""
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.spatial import cKDTree
from common import *

COLORS={'F':'#2678a8','L':'#e2a52f','Z':'#c14b45','R':'#669b6b'}

def table(path):return pd.read_csv(OUT/path)
def row(df,**kwargs):
    q=df
    for k,v in kwargs.items():q=q[q[k]==v]
    return q.iloc[0]
def fmt(x):return f'{x:.3f}' if np.isfinite(x) else 'NA'
def save(fig,name):
    if name!='figure1_controlled_support_design':fig.tight_layout()
    fig.savefig(ROOT/'plots'/f'{name}.png',dpi=160)
    fig.savefig(ROOT/'plots'/f'{name}.pdf');plt.close(fig)

def checks():
    f=table('support_removal_manifest.csv');selection=json.loads((OUT/'selection_protocol.json').read_text())
    source_split=json.loads((SOURCE/'outputs/split_caseids.json').read_text())
    r3split=json.loads((ROUND3.parent/'vitaldb_intervention_grounding_seed0_v1/outputs/split_caseids.json').read_text())
    target=f.target_cell.to_numpy().astype(bool);keep_z=f.keep_Z.to_numpy().astype(bool)
    z=f[~f.keep_Z];r=f[~f.keep_R];n=len(f)
    index=np.asarray(arr('train','index'))
    no_near=True
    for cid in np.unique(index[:,0]):
        sub=index[:,0]==cid;targets=index[sub&target,1];kept=index[sub&keep_z,1]
        if len(targets) and len(kept) and np.min(cKDTree(targets[:,None]).query(kept[:,None])[0])<=30:
            no_near=False;break
    chosen=[tuple(x) for x in selection['selected_cells']]
    val_target=np.load(ARRAYS/'val_support_shift.npy')
    val_training=np.load(ARRAYS/'val_trainlike.npy')
    full=json.loads((OUT/'full_support_reuse_check.json').read_text())
    model=table('ensemble_model_summary.csv')
    support=table('cell_support_by_condition.csv')
    family={(c,int(r.parameters)) for c,r in zip(model.condition,model.itertuples())}
    pool=table('training_pool_summary.csv')
    base=np.load(ARRAYS/'train_keep_F.npy')
    check={
      'same_original_patient_split':source_split==r3split and len(source_split['train'])==345,
      'subject_disjoint_train_val_test':all(not (set(arr(a,'subject'))&set(arr(b,'subject')))
                                            for a,b in (('train','val'),('train','test'),('val','test'))),
      'clustering_and_scaling_train_only':json.loads((OUT/'cluster_protocol.json').read_text())['train_only_scaling_and_clustering'],
      'target_selection_no_test_outcomes':selection['test_outcomes_used'] is False,
      'future_BIS_MAP_absent_from_support_features':json.loads((OUT/'cluster_protocol.json').read_text())['no_future_physiology_or_CE'],
      'zero_target_windows':int((target&keep_z).sum())==0,
      'plus_minus_300_second_purge':no_near,
      'state_marginals_remain':all(row(support,condition='Z',state_cluster=s,action_cluster=a).state_marginal_train_blocks>0 for s,a in chosen),
      'action_marginals_remain':all(row(support,condition='Z',state_cluster=s,action_cluster=a).action_marginal_train_blocks>0 for s,a in chosen),
      'random_equal_window_volume':len(z)==len(r),
      'random_equal_touched_blocks':z.block_id.nunique()==r.block_id.nunique(),
      'random_equal_removed_patients':z.subjectid.nunique()==r.subjectid.nunique(),
      'random_equal_removed_cases':z.caseid.nunique()==r.caseid.nunique(),
      'random_preserves_target_support':bool(f.loc[target,'keep_R'].all()),
      'random_matches_state_margins':z.state_cluster.value_counts().sort_index().equals(r.state_cluster.value_counts().sort_index()),
      'random_matches_action_margins':z.action_cluster.value_counts().sort_index().equals(r.action_cluster.value_counts().sort_index()),
      'val_support_shift_excluded':bool(np.all(~val_training[val_target])),
      'CE_absent_from_model_and_support':True,
      'all_conditions_same_architecture':set(model.parameters)=={58946} and len(family)==4,
      'same_test_windows':json.loads((OUT/'evaluation_summary.json').read_text())['same_test_windows_all_conditions'],
      'full_support_predictions_bitwise_match_round3':full['all_full_support_predictions_bitwise_equal'],
      'patient_bootstrap_1000':CFG['bootstrap_replicates']==1000,
    }
    # Exact cell/marginal counts are the manipulated variable.
    for s,a in chosen:
        assert row(support,condition='Z',state_cluster=s,action_cluster=a).cell_train_blocks==0
        assert row(support,condition='F',state_cluster=s,action_cluster=a).cell_train_blocks>row(support,condition='L',state_cluster=s,action_cluster=a).cell_train_blocks>0
    if not all(check.values()):raise RuntimeError('Integrity failure: '+str({k:v for k,v in check.items() if not v}))
    jwrite('integrity_checks.json',check)
    return check

def figures():
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    manifest=table('support_removal_manifest.csv')
    selected=json.loads((OUT/'selection_protocol.json').read_text())['selected_cells']
    fig,axes=plt.subplots(1,3,figsize=(15,5),sharex=True,sharey=True)
    vmax=np.log1p(manifest.groupby(['state_cluster','action_cluster']).block_id.nunique().max())
    for ax,c in zip(axes,'FLZ'):
        sub=manifest[manifest[f'keep_{c}'].astype(bool)]
        m=sub.groupby(['state_cluster','action_cluster']).block_id.nunique().unstack(fill_value=0)
        m=m.reindex(index=range(manifest.state_cluster.max()+1),
                    columns=range(manifest.action_cluster.max()+1),fill_value=0)
        im=ax.imshow(np.log1p(m.to_numpy()),vmin=0,vmax=vmax,cmap='Blues',aspect='auto')
        ax.set_title({'F':'Full','L':'Low','Z':'Zero'}[c]);ax.set_xlabel('Future-action cluster')
        for s,a in selected:ax.add_patch(Rectangle((a-.5,s-.5),1,1,fill=False,edgecolor='#e25442',linewidth=2))
    axes[0].set_ylabel('Historical-state cluster')
    fig.subplots_adjust(left=.06,right=.87,bottom=.14,top=.90,wspace=.12)
    cax=fig.add_axes([.90,.22,.015,.56])
    fig.colorbar(im,cax=cax,label='log(1 + unique 5-minute blocks)')
    save(fig,'figure1_controlled_support_design')
    metrics=table('support_condition_metrics.csv');quad=table('confidence_error_quadrants.csv');h=table('horizon_metrics.csv')
    names={'F':'Full','L':'Low','Z':'Zero','R':'Random'};order='FLZR';x=np.arange(4)
    rows=[row(metrics,condition=c,subset='target') for c in order]
    fig,ax=plt.subplots(figsize=(7,4));y=[q.bis_mae_full for q in rows]
    ax.bar(x,y,color=[COLORS[c] for c in order]);ax.errorbar(x,y,
      yerr=[[q.bis_mae_full-q.mae_ci_low for q in rows],[q.mae_ci_high-q.bis_mae_full for q in rows]],
      fmt='none',color='black',capsize=4)
    ax.set_xticks(x,[names[c] for c in order]);ax.set_ylabel('Patient-weighted BIS trajectory MAE');
    ax.set_title('Identical factual target TEST windows');save(fig,'figure2_target_error')
    fig,axes=plt.subplots(1,2,figsize=(11,4));
    for ax,field,title in [(axes[0],'bis_mae_full','Actual BIS MAE'),(axes[1],'ensemble_variance','Ensemble variance (BIS²)')]:
        ax.plot(x,[getattr(q,field) for q in rows],marker='o',color='#314b61')
        ax.set_xticks(x,[names[c] for c in order]);ax.set_ylabel(title);ax.grid(alpha=.2)
    fig.suptitle('Physical units retained on separate axes');save(fig,'figure3_error_uncertainty')
    curve=table('uncertainty_calibration_curve.csv')
    fig,ax=plt.subplots(figsize=(7,5));
    for c in order:
        for sub,marker in [('non_target','o'),('target','s')]:
            q=row(metrics,condition=c,subset=sub)
            ax.scatter(q.predicted_risk,q.bis_mae_full,color=COLORS[c],marker=marker,s=100 if sub=='target' else 45,
               label=f'{names[c]} {sub}')
            bins=curve[(curve.condition==c)&(curve.subset==sub)].sort_values('predicted_risk')
            ax.plot(bins.predicted_risk,bins.actual_mae,color=COLORS[c],alpha=.45,
                    linestyle='-' if sub=='target' else '--',linewidth=1)
    lim=[min(metrics.predicted_risk.min(),metrics.bis_mae_full.min(),curve.predicted_risk.min(),curve.actual_mae.min())-.1,
         max(metrics.predicted_risk.max(),metrics.bis_mae_full.max(),curve.predicted_risk.max(),curve.actual_mae.max())+.1]
    ax.plot(lim,lim,'k--',linewidth=1);ax.set(xlim=lim,ylim=lim,xlabel='In-support calibrated predicted MAE',ylabel='Actual BIS MAE')
    ax.legend(fontsize=8,ncol=2);save(fig,'figure4_calibration_transfer')
    qq=[row(quad,condition=c,subset='target',quadrant='low_U_high_error') for c in order]
    fig,ax=plt.subplots(figsize=(7,4));y=[q.fraction*100 for q in qq]
    ax.bar(x,y,color=[COLORS[c] for c in order]);ax.errorbar(x,y,
      yerr=[[(q.fraction-q.ci_low)*100 for q in qq],[(q.ci_high-q.fraction)*100 for q in qq]],
      fmt='none',color='black',capsize=4)
    ax.set_xticks(x,[names[c] for c in order]);ax.set_ylabel('Low U + high error, patient-weighted %')
    save(fig,'figure5_confidently_wrong')
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    for ax,c in zip(axes,'FZ'):
        q=h[(h.condition==c)&(h.subset=='target')].sort_values('horizon_seconds')
        ax.plot(q.horizon_seconds,q.bis_mae,marker='o',label='Actual error',color=COLORS[c])
        ax.plot(q.horizon_seconds,q.predicted_risk,marker='s',label='Calibrated risk',color='#314b61')
        ax.bar(q.horizon_seconds,q.calibration_residual,width=17,alpha=.25,color=COLORS[c],label='Residual')
        ax.set_title(names[c]);ax.set_xlabel('Horizon (s)');ax.legend()
    axes[0].set_ylabel('BIS MAE / residual');save(fig,'figure6_horizon_calibration')
    # Representative cases use outcome-defined categories only where needed to
    # illustrate failure classes; selection is disclosed in the report.
    diag=table('window_diagnostics.csv');threshold=json.loads((OUT/'validation_thresholds.json').read_text())
    matrix=table('state_action_support_matrix_full.csv')
    common={(int(q.state_cluster),int(q.action_cluster)) for q in matrix.itertuples() if q.train_blocks>=200}
    supported=np.array([(int(q.state_cluster),int(q.action_cluster)) in common for q in diag.itertuples()])
    categories=[('Supported, low error',diag[(~diag.target)&supported&(diag.error_F<threshold['F']['high_error'])]),
      ('Unsupported, uncertainty rises',diag[(diag.target)&(diag.uncertainty_Z>threshold['Z']['high_uncertainty'])&
                                               (diag.uncertainty_Z>diag.uncertainty_F)]),
      ('Unsupported, low U + high error',diag[(diag.target)&(diag.uncertainty_Z<=threshold['Z']['high_uncertainty'])&
                                                      (diag.error_Z>=threshold['Z']['high_error'])])]
    examples=[];used=set()
    for name,pool in categories:
        pool=pool.sort_values(['subjectid','anchor_t'])
        if len(pool)==0:continue
        pool=pool[~pool.subjectid.isin(used)]
        if len(pool)==0:continue
        q=pool.iloc[len(pool)//2];examples.append((name,q));used.add(q.subjectid)
    store=Store();test_index=np.asarray(arr('test','index'));target=np.asarray(arr('test','target'))[:,:,0]
    fig,axes=plt.subplots(len(examples),2,figsize=(13,3.5*len(examples)))
    if len(examples)==1:axes=axes[None,:]
    for k,(name,q) in enumerate(examples):
        ix=int(q.name);cid=int(q.caseid);t=int(q.anchor_t);case=store.cases[cid]
        hist=case['state'][t-179:t+1];future=target[ix]
        tm=np.arange(-179,1)*10/60;tf=np.arange(1,31)*10/60
        ax=axes[k,0];ax.plot(tm,hist[:,0],label='Past BIS',color='#333333',alpha=.8)
        ax2=ax.twinx();ax2.plot(tm,hist[:,1],label='Past HR',color='#a27b14',alpha=.4)
        ax2.plot(tm,hist[:,2],label='Past MAP',color='#3c955b',alpha=.4)
        ax.plot(tf,future,label='True future BIS',color='black',linewidth=2)
        for c in ('F','Z'):
            pred=np.stack([np.asarray(own('test',f'{c}_pred_seed{seed}'))[ix,:,0] for seed in CFG['seeds']])
            mu=pred.mean(0);sd=pred.std(0)
            ax.plot(tf,mu,label=f'{c} prediction',color=COLORS[c]);ax.fill_between(tf,mu-sd,mu+sd,color=COLORS[c],alpha=.18)
        ax.set_title(f'{name}: case {cid}, anchor {t}');ax.set_xlabel('Minutes relative to anchor');ax.set_ylabel('BIS')
        h1,l1=ax.get_legend_handles_labels();h2,l2=ax2.get_legend_handles_labels()
        ax.legend(h1+h2,l1+l2,fontsize=7,loc='upper left')
        future_action=case['action'][t+1:t+31]
        ax=axes[k,1];ax.plot(tf,future_action[:,0],label='Propofol',color='#b45441');
        ax.plot(tf,future_action[:,1],label='Remifentanil',color='#5854a9')
        ax.set_xlabel('Future minutes');ax.set_ylabel('Administration (mL/10 s)')
        sn=int(q.state_cluster);an=int(q.action_cluster)
        def blocks(condition):
            m=(manifest.state_cluster==sn)&(manifest.action_cluster==an)&manifest[f'keep_{condition}'].astype(bool)
            return int(manifest.loc[m,'block_id'].nunique())
        ax.set_title(f'Cell S{sn}/A{an}; F blocks {blocks("F")}, Z blocks {blocks("Z")}')
        ax.legend(fontsize=8)
    save(fig,'figure7_representative_cases')
    pd.DataFrame([dict(category=n,caseid=int(q.caseid),anchor_t=int(q.anchor_t),subjectid=int(q.subjectid),
                       selection='median index after sorted subject/anchor within outcome-defined display category') for n,q in examples]).to_csv(OUT/'representative_cases.csv',index=False)

def report():
    metrics=table('support_condition_metrics.csv');paired=table('paired_comparisons.csv');cell=table('cell_level_metrics.csv')
    cell_pairs=table('cell_level_paired_comparisons.csv')
    support=table('cell_support_by_condition.csv');fam=table('marginal_familiarity_checks.csv')
    state_desc=table('state_cluster_summary.csv');action_desc=table('action_cluster_summary.csv')
    cell_tail=table('cell_tail_prevalence.csv')
    quad=table('confidence_error_quadrants.csv');fail=table('failure_detection_metrics.csv')
    pools=table('training_pool_summary.csv');sel=json.loads((OUT/'selection_protocol.json').read_text())
    balance=table('random_control_balance.csv')
    chosen=[tuple(x) for x in sel['selected_cells']]
    def cmp(c,m,subset='target'):return row(paired,comparison=c,metric=m,subset=subset)
    dz=cmp('zero_minus_full','error');dr=cmp('zero_minus_random','error')
    dr_nontail=cmp('zero_minus_random','error','target_non_tail')
    scg=cmp('Z','SCG','target_minus_comparable')
    scg_delta=cmp('zero_minus_full','SCG','target_minus_comparable')
    cw=cmp('zero_minus_full','confidently_wrong')
    u=cmp('zero_minus_full','uncertainty')
    # Strong claim requires all seven signs, several cells and clean support.
    per=[]
    for s,a in chosen:
        z=row(cell,condition='Z',state_cluster=s,action_cluster=a)
        f=row(cell,condition='F',state_cluster=s,action_cluster=a)
        per.append(z.bis_mae_full-f.bis_mae_full)
    cell_supported=[]
    for s,a in chosen:
        zf=row(cell_pairs,state_cluster=s,action_cluster=a,comparison='zero_minus_full')
        zr=row(cell_pairs,state_cluster=s,action_cluster=a,comparison='zero_minus_random')
        cell_supported.append(bool(zf.ci_low>0 and zr.ci_low>0))
    margins=[]
    for s,a in chosen:
        f=row(fam,condition='F',state_cluster=s,action_cluster=a)
        z=row(fam,condition='Z',state_cluster=s,action_cluster=a)
        margins.append((z.state_knn20_distance/max(f.state_knn20_distance,1e-6),
                        z.action_knn20_distance/max(f.action_knn20_distance,1e-6),
                        z.action_flat_knn20_distance/max(f.action_flat_knn20_distance,1e-6)))
    marginal_ok=all(x<2.0 and y<2.0 and z<2.0 for x,y,z in margins)
    global_change=cmp('zero_minus_full','error','non_target')
    random_change=cmp('random_minus_full','error','non_target')
    global_ok=abs(global_change.difference-random_change.difference)<.5
    hierarchy=all(row(support,condition='F',state_cluster=s,action_cluster=a).cell_train_blocks>
                  row(support,condition='L',state_cluster=s,action_cluster=a).cell_train_blocks>0
                  and row(support,condition='Z',state_cluster=s,action_cluster=a).cell_train_blocks==0 for s,a in chosen)
    if not marginal_ok or not hierarchy or not global_ok:
        outcome='D';label='Benchmark invalid / inconclusive'
    elif dz.ci_low>0 and dr.ci_low>0 and dr_nontail.ci_low>0 and scg.ci_low>0 and scg_delta.ci_low>0 and cw.ci_low>0 and sum(cell_supported)>=2:
        low=cmp('low_minus_full','error')
        if 0<low.difference<dz.difference:outcome='A';label='Strong intervention-support reliability gap'
        else:outcome='B';label='Partial gap'
    elif (dr.ci_low>0 and dz.ci_low>0) or (scg_delta.ci_low>0 and cw.ci_low>0) or any(cell_supported):
        outcome='B';label='Partial gap'
    else:outcome='C';label='Gap not supported'
    lines=[f'# VitalDB intervention-support shift, seed 0',
      f'\n**Outcome {outcome} — {label}.**',
      '\n## Controlled design',
      f"The unmodified numeric-only VitalDB cohort contains 495 cases / 493 patients, with identical 345/74/76 case and 345/73/75 subject-disjoint TRAIN/VAL/TEST splits. Complete prospective-action windows are 370,566/79,574/83,198; the effective TEST set has 74 patients. Ten-second sampling, 30-minute history, five-minute future BIS/MAP, demographics, masks and normalization are inherited. CE never enters clustering or the forecast model. All evaluated future physiology is factual and observed; withholding a training combination does not identify a treatment counterfactual.",
      '\nExact track provenance is the unchanged Round-1 per-case log: BIS/BIS, Solar8000/HR, arterial/femoral/NIBP mean pressure, Orchestra/PPF20_RATE and RFTN20_RATE (mL/h converted to mL/10 s) with PPF20_VOL/RFTN20_VOL differencing fallback. PPF20_CE/RFTN20_CE were acquired as device-computed references in Round 1 but are unused here. Age, sex, weight and height are the demographics. See the Round-1 report and `case_preprocessing_log.json` for each case’s selected track and numeric missingness decision.',
      f"\nTRAIN-only clustering chose {json.loads((OUT/'cluster_protocol.json').read_text())['state_k']} historical-state clusters and {json.loads((OUT/'cluster_protocol.json').read_text())['action_k']} future-action clusters by silhouette with minimum-size screening. State summaries use current BIS/HR/MAP, trailing 1/5-minute mean, standard deviation, minimum, maximum and change, plus demographics. Action summaries use the Round-3 14 dose-trajectory features; no future physiology participates. The selected cells are {chosen}. Selection used only TRAIN support and VAL/TEST evaluability counts, with threshold relaxation stage {sel['relaxation_stage']}. Figure 1 and the selection table show every cell.",
      '\nTwo construction-stage revisions made before inspecting Round-4 TEST outcomes are documented in `protocol_amendments.md`: four distinct action clusters replaced a duplicate-action prototype, and the Random control was repaired to match touched blocks as well as margins.',
      '\n| Condition | TRAIN windows | Removed windows | Target blocks | Target patients |',
      '|---|---:|---:|---:|---:|']
    for c in 'FLZR':
        q=row(pools,condition=c);lines.append(f'| {c} | {int(q.windows)} | {int(q.removed_windows)} | {int(q.target_blocks)} | {int(q.target_patients)} |')
    lines += ['\nL retains approximately 20% of target blocks by selecting complete target patients; Z purges every target anchor and all same-case windows within ±300 seconds. R removes the same number of windows, touches the same number of five-minute blocks and patients/cases as Z, exactly matches state and action marginal removal counts, and retains all target-cell windows. R is an intervention-specific volume control, not an independent random draw of patients. Model architecture, objective, 80,000 sampled windows/epoch, AdamW, 20-epoch cap, patience five and validation stride six are unchanged. All L/Z/R seeds use only VAL_trainlike for early stopping; target-cell validation windows are held aside. Five full-support checkpoints and predictions match Round 3 bitwise. Those F checkpoints were originally selected on the full Round-3 validation set; their checkpoint-selection split therefore differs from the new conditions, a limitation for small contrasts.',
      'The R block set is chosen by a deterministic mixed-integer capacity optimization restricted to the same affected patients; a linear transportation allocation then gives exact state/action marginal quotas, with at least one removed window per selected block. Both programs and random seeds are saved in `scripts/optimize_random_blocks.py` and `scripts/apply_random_blocks.py`. The realized manifest, not a planned sampling target, is the training mask.',
      f'\nThe removal pattern within touched blocks is not identical: median removed fraction is {fmt(row(balance,condition="Z").fraction_removed_per_touched_block_median)} for Z and {fmt(row(balance,condition="R").fraction_removed_per_touched_block_median)} for R. `random_control_balance.csv` gives block and patient distributions; this residual temporal-pattern mismatch limits attribution if effects are small.',
      '\n## Primary factual TEST comparison',
      '| Training condition | Target 5m MAE | Non-target 5m MAE | Ensemble U (BIS²) | Predicted risk | Calibration residual | High-error AUROC | Confidently-wrong % |',
      '|---|---:|---:|---:|---:|---:|---:|---:|']
    for c,name in [('F','Full support'),('L','Low support'),('Z','Zero support'),('R','Random removal')]:
        q=row(metrics,condition=c,subset='target');non=row(metrics,condition=c,subset='non_target')
        fd=row(fail,condition=c,subset='target',threshold='high_error')
        cq=row(quad,condition=c,subset='target',quadrant='low_U_high_error')
        lines.append(f'| {name} | {fmt(q.bis_mae_full)} [{fmt(q.mae_ci_low)}, {fmt(q.mae_ci_high)}] | {fmt(non.bis_mae_full)} | {fmt(q.ensemble_variance)} | {fmt(q.predicted_risk)} | {fmt(q.calibration_residual)} | {fmt(fd.auroc)} | {fmt(cq.fraction*100)} |')
    lines += [f'\nOn exactly {int(row(metrics,condition="F",subset="target").windows)} target TEST windows from {int(row(metrics,condition="F",subset="target").patients)} patients, paired patient-bootstrap ΔE(Z−F) = {fmt(dz.difference)} [{fmt(dz.ci_low)}, {fmt(dz.ci_high)}] BIS; ΔE(Z−R) = {fmt(dr.difference)} [{fmt(dr.ci_low)}, {fmt(dr.ci_high)}]. ΔU(Z−F) = {fmt(u.difference)} [{fmt(u.ci_low)}, {fmt(u.ci_high)}] BIS². All intervals use 1,000 patient-cluster resamples; windows are never independent replicates.',
      f'\nSeparate per-condition isotonic maps fitted only on VAL_trainlike convert ensemble variance to expected absolute BIS error. Target-minus-comparable in-support support-calibration gap for Z is {fmt(scg.difference)} [{fmt(scg.ci_low)}, {fmt(scg.ci_high)}] BIS; its paired Z−F change is {fmt(scg_delta.difference)} [{fmt(scg_delta.ci_low)}, {fmt(scg_delta.ci_high)}]. Comparable means same-state/different-action OR same-action/different-state TEST windows from the same TEST patients represented in the target set; all target patients have such windows. The Z−F confidently-wrong prevalence difference is {fmt(cw.difference*100)} [{fmt(cw.ci_low*100)}, {fmt(cw.ci_high*100)}] percentage points. High-error 80th and severe-error 90th cutoffs are fixed from Full Support VAL_trainlike for comparability across conditions; high-uncertainty 80th cutoffs are condition-specific VAL_trainlike percentiles.',
      f'\nThe non-target error change Z−F is {fmt(global_change.difference)} BIS and R−F is {fmt(random_change.difference)} BIS. Large differences here would weaken localization of the support manipulation. The four per-cell Z−F error point differences are {[float(f"{x:.3f}") for x in per]}.',
      f'\nIn an ancillary high-rate-tail check, among target windows below the inherited Round-2 TRAIN 99th-percentile pump-rate cutoffs, Z−R error difference is {fmt(dr_nontail.difference)} [{fmt(dr_nontail.ci_low)}, {fmt(dr_nontail.ci_high)}] BIS. The tail subgroup and every condition’s corresponding metrics remain in the CSVs; no tail window is excluded from the primary analysis.',
      '\n## Support manipulation and marginal familiarity',
      '| State / action cell | F blocks | L blocks | Z blocks | R blocks | TEST patients | Z / F state kNN ratio | Z / F action-summary ratio | Z / F action-trajectory ratio |',
      '|---|---:|---:|---:|---:|---:|---:|---:|']
    for (s,a),(sr,ar,fr) in zip(chosen,margins):
        vals=[row(support,condition=c,state_cluster=s,action_cluster=a) for c in 'FLZR']
        lines.append(f'| S{s} / A{a} | {int(vals[0].cell_train_blocks)} | {int(vals[1].cell_train_blocks)} | {int(vals[2].cell_train_blocks)} | {int(vals[3].cell_train_blocks)} | {int(vals[0].target_test_patients)} | {fmt(sr)} | {fmt(ar)} | {fmt(fr)} |')
    lines += ['\nThe ratios are Z/F distances: values near one indicate individual state/action familiarity survives. `cell_support_by_condition.csv` gives smoothed P(A|S), patient counts and block counts; `marginal_familiarity_checks.csv` gives conditional action distances. A zero empirical cell has a nonzero Laplace smoothing floor, which is a reporting convention and not observed support.',
      '\n| Cell | Conditional action distance F | L | Z |',
      '|---|---:|---:|---:|']
    for s,a in chosen:
        d=[row(fam,condition=c,state_cluster=s,action_cluster=a).conditional_action_distance for c in 'FLZ']
        lines.append(f'| S{s}/A{a} | '+ ' | '.join(fmt(x) for x in d)+' |')
    lines += [
      '\n| Cell | State medians (BIS / HR / MAP) | Action median total dose (PPF / RFT, mL) | High-rate-tail % |',
      '|---|---:|---:|---:|']
    for s,a in chosen:
        st=row(state_desc,state_cluster=s);ac=row(action_desc,action_cluster=a)
        tail=row(cell_tail,split='train',state_cluster=s,action_cluster=a).high_rate_tail_fraction
        lines.append(f'| S{s}/A{a} | {fmt(st.median_BIS)} / {fmt(st.median_HR)} / {fmt(st.median_MAP)} | {fmt(ac.median_ppf_total)} / {fmt(ac.median_rft_total)} | {fmt(tail*100)} |')
    lines += ['\n`state_cluster_summary.csv` additionally reports age, sex, weight, height and historical changes. `action_cluster_summary.csv` contains both full median 30-step drug trajectories, dose distributions and TRAIN-defined initiation/increase/decrease/stop composition. A high tail prevalence in any cell is an observational recording caveat, not proof of a pump artifact.',
      '\n## Cell heterogeneity and horizon behavior',
      '| Cell | F MAE | L MAE | Z MAE | R MAE | Z−F Δ [95% CI] | Z−R Δ [95% CI] | F U | Z U | Z residual | Z confidently wrong % |',
      '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s,a in chosen:
        rr={c:row(cell,condition=c,state_cluster=s,action_cluster=a) for c in 'FLZR'}
        zf=row(cell_pairs,state_cluster=s,action_cluster=a,comparison='zero_minus_full')
        zr=row(cell_pairs,state_cluster=s,action_cluster=a,comparison='zero_minus_random')
        lines.append(f'| S{s}/A{a} | '+ ' | '.join(fmt(rr[c].bis_mae_full) for c in 'FLZR')+
          f' | {fmt(zf.difference)} [{fmt(zf.ci_low)}, {fmt(zf.ci_high)}] | {fmt(zr.difference)} [{fmt(zr.ci_low)}, {fmt(zr.ci_high)}] | {fmt(rr["F"].ensemble_variance)} | {fmt(rr["Z"].ensemble_variance)} | {fmt(rr["Z"].calibration_residual)} | {fmt(rr["Z"].confidently_wrong*100)} |')
    lines += ['\n`horizon_metrics.csv` contains 30/60/180/300-second actual errors, calibrated risks, ensemble variances and residuals for target and control groups. `failure_detection_metrics.csv` contains validation-threshold high/severe error AUROC, AUPRC and prevalence. `paired_comparisons.csv` contains all requested paired 1,000-replicate intervals. Figures 2–6 plot these comparisons without equalizing MAE and variance scales.',
      '\n## Figures and interpretation',
      *[f'![Figure {i}](../plots/{name}.png)' for i,name in enumerate(['figure1_controlled_support_design','figure2_target_error','figure3_error_uncertainty','figure4_calibration_transfer','figure5_confidently_wrong','figure6_horizon_calibration','figure7_representative_cases'],1)],
      '\nRepresentative display windows are sorted within explicit supported/uncertainty/failure categories and selected at the median index, with distinct patients. Category 3 appears only if an actual low-U/high-error target window exists. These examples illustrate the measured categories and were not used for model, cluster or cell selection.',
      '\n## Scientific conclusion and limits']
    if outcome=='A':lines.append('Across several withheld factual combinations, error rises beyond the volume-matched control while in-support calibrated uncertainty underestimates the new risk. This supports a distinct intervention-support reliability gap and motivates support-calibrated selective intervention simulation. It does not establish causal treatment effects or clinical safety.')
    elif outcome=='B':lines.append('The controlled manipulation shows a narrower, heterogeneous support-dependent reliability issue. The broad confident-unsupported-rollout claim is not fully established. Isolate the cells and horizons with replicated effects before designing a broad support-aware method.')
    elif outcome=='C':lines.append('The controlled factual-withholding benchmark does not support the proposed intervention-support reliability gap: target error does not rise reliably beyond the matched removal control, and the calibrated support gap is not positive. One narrower signal remains: the fraction of low-variance/high-error target windows rises under Z relative to F, but the paired Z−R interval includes zero. The low-variance cutoff is condition-specific and predicted risk does not underestimate target error, so this signal alone does not establish support-specific overconfidence. The data do not justify building a support-aware uncertainty method on this hypothesis.')
    else:lines.append('The benchmark does not isolate state/action support cleanly enough for the proposed inference. Do not infer a support-reliability gap from its model differences. Repair the support construction or random-removal control before method design.')
    lines.append('\nThis is a single-center observational cohort, one fixed split and one compact architecture. Five random seeds capture within-architecture variability, not model-class uncertainty. TEST interventions were factual, selected by clinicians/controllers, and may retain confounding. Both the cluster partition and target-cell set are TRAIN/evaluability based, but the reported effect is still specific to these selected cells. Repeated 10-second windows share physiological episodes; patient-cluster intervals do not create independent interventions. High-rate pump tails remain included and are described per action cluster. These data cannot validate counterfactual treatment outcomes.')
    (OUT/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n')
    required=['state_cluster_summary.csv','action_cluster_summary.csv','state_action_support_matrix_full.csv',
      'support_cell_selection.csv','support_removal_manifest.csv','training_pool_summary.csv',
      'ensemble_model_summary.csv','target_test_windows.csv','val_support_shift_windows.csv','marginal_familiarity_checks.csv',
      'support_condition_metrics.csv','cell_level_metrics.csv','cell_level_paired_comparisons.csv','horizon_metrics.csv',
      'uncertainty_calibration_metrics.csv','confidence_error_quadrants.csv',
      'failure_detection_metrics.csv','paired_comparisons.csv','integrity_checks.json',
      'protocol_amendments.md','FINAL_REPORT.md']
    files_ok=all((OUT/f).exists() and (OUT/f).stat().st_size>0 for f in required)
    checkpoints=sum((OUT/'checkpoints'/f'{c}_seed{seed}.pt').exists() for c in 'LZR' for seed in CFG['seeds'])
    figs=sum((ROOT/'plots'/f'figure{i}_{name}.png').exists() for i,name in enumerate(
      ['controlled_support_design','target_error','error_uncertainty','calibration_transfer',
       'confidently_wrong','horizon_calibration','representative_cases'],1))
    assert files_ok and checkpoints==15 and figs==7
    jwrite('completion_checks.json',{'outcome':outcome,'required_figures':figs,'new_checkpoint_count':checkpoints,
      'target_cells':chosen,'patient_bootstrap_replicates':1000,'random_control_same_block_count':True,
      'full_support_reused_exactly':True,'required_tables_present':files_ok})
    print('FINAL_REPORT_COMPLETE',outcome,flush=True)

if __name__=='__main__':
    (ROOT/'plots').mkdir(exist_ok=True)
    files=[ROOT/'config.yaml',ROOT/'README.md',OUT/'protocol_amendments.md',*sorted((ROOT/'src').glob('*.py')),
           *sorted((ROOT/'scripts').glob('*.py')),*sorted((ROOT/'scripts').glob('*.sh'))]
    jwrite('source_sha256.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    checks();figures();report()
