"""Patient-weighted factual failure, detection, coverage, horizon and subgroup audit."""
import json,warnings
import numpy as np
import pandas as pd
from scipy.stats import rankdata,pearsonr,spearmanr
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from common import *

SIGNALS=['U_ensemble','D_raw_physiology','D_state_knn','D_state_mahalanobis',
         'D_action','D_action_summary','D_action_rarity','D_joint_knn',
         'D_joint_mahalanobis','D_cond','D_perp','D_random_latent','D_random_joint']
GEOMETRY=['D_state_knn','D_state_mahalanobis','D_action','D_joint_knn',
          'D_joint_mahalanobis','D_cond','D_perp']
REPS=CFG['bootstrap_replicates']

def patient_index(subject):
    ids,inv,count=np.unique(subject,return_inverse=True,return_counts=True)
    return ids,inv,count,1/count[inv]

def boot_draws(n,reps=REPS):
    rng=np.random.default_rng(0)
    return np.array([np.bincount(rng.integers(n,size=n),minlength=n) for _ in range(reps)],dtype='float32')

def weighted_corr(x,y,w):
    w=w/w.sum();mx=np.sum(w*x);my=np.sum(w*y)
    cov=np.sum(w*(x-mx)*(y-my));vx=np.sum(w*(x-mx)**2);vy=np.sum(w*(y-my)**2)
    return float(cov/np.sqrt(vx*vy)) if vx>0 and vy>0 else np.nan

def corr_boot(x,y,subject,draws,rank=False):
    valid=np.isfinite(x)&np.isfinite(y);x=np.asarray(x)[valid];y=np.asarray(y)[valid]
    ids,inv,count,w=patient_index(np.asarray(subject)[valid]);assert len(ids)==draws.shape[1]
    if rank:x=rankdata(x).astype('float64');y=rankdata(y).astype('float64')
    else:x=x.astype('float64');y=y.astype('float64')
    # Per-patient sufficient statistics make 1,000 cluster replicates exact for
    # the fixed pooled ranks, while preserving equal patient mass.
    moments=np.stack([np.bincount(inv,weights=w*v,minlength=len(ids)) for v in
        [np.ones(len(x)),x,y,x*x,y*y,x*y]],1)
    mx=draws@moments;means=mx[:,1:]/mx[:,[0]]
    cov=mx[:,5]/mx[:,0]-means[:,0]*means[:,1]
    vx=mx[:,3]/mx[:,0]-means[:,0]**2;vy=mx[:,4]/mx[:,0]-means[:,1]**2
    boot=cov/np.sqrt(np.maximum(vx*vy,1e-20))
    est=weighted_corr(x,y,w)
    return est,float(np.quantile(boot,.025)),float(np.quantile(boot,.975)),boot

def auc_ap_boot(score,label,subject,draws,ci=True):
    valid=np.isfinite(score)&np.isfinite(label)
    s=np.asarray(score)[valid];y=np.asarray(label)[valid].astype(bool);sub=np.asarray(subject)[valid]
    ids,inv,count,w=patient_index(sub);assert len(ids)==draws.shape[1]
    auc=float(roc_auc_score(y,s,sample_weight=w));ap=float(average_precision_score(y,s,sample_weight=w))
    if not ci:return auc,np.nan,np.nan,ap,np.nan,np.nan,None,None
    order=np.argsort(s,kind='stable')[::-1];so=s[order];yo=y[order];io=inv[order];wo=w[order]
    # Aggregate ties at their right endpoint so AP and AUROC match sklearn.
    ends=np.r_[np.flatnonzero(so[1:]!=so[:-1]),len(so)-1]
    starts=np.r_[0,ends[:-1]+1]
    au=[];aps=[]
    for mult in draws:
        mass=wo*mult[io]
        pos=np.add.reduceat(mass*yo,starts);neg=np.add.reduceat(mass*(~yo),starts)
        cp=np.cumsum(pos);cn=np.cumsum(neg);P=cp[-1];N=cn[-1]
        if P<=0 or N<=0:au.append(np.nan);aps.append(np.nan);continue
        au.append(np.sum(pos*(N-cn+neg/2))/(P*N))
        aps.append(np.sum((cp/(cp+cn+1e-12))*pos)/P)
    au=np.asarray(au);aps=np.asarray(aps)
    return auc,*np.nanquantile(au,[.025,.975]),ap,*np.nanquantile(aps,[.025,.975]),au,aps

def error_arrays(split):
    ps=np.stack([np.load(ARRAYS/f'{split}_pred_seed{s}.npy',mmap_mode='r')[:,:,0] for s in CFG['seeds']])
    mean=ps.mean(0);var=ps.var(0,ddof=0)
    truth=np.load(ARRAYS/f'{split}_target.npy',mmap_mode='r')[:,:,0]
    mask=np.load(ARRAYS/f'{split}_mask.npy',mmap_mode='r')
    err=np.where(mask,np.abs(mean-truth),np.nan)
    sq=np.where(mask,(mean-truth)**2,np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore');full=np.nanmean(err,1);full_sq=np.nanmean(sq,1)
    return mean,var,err,sq,full,full_sq

def coverage_rows(scores,error,sq,subject,draws):
    ids,inv,count,w=patient_index(subject)
    cover=CFG['coverage'];rows=[];patient_risk={}
    for name in scores.columns:
        value=scores[name].to_numpy();valid=np.isfinite(value)&np.isfinite(error)
        order=np.flatnonzero(valid)[np.argsort(value[valid],kind='stable')]
        for c in cover:
            take=order[:max(1,int(round(c*len(order))))]
            cnt=np.bincount(inv[take],minlength=len(ids))
            pe=np.divide(np.bincount(inv[take],weights=error[take],minlength=len(ids)),cnt,
                         out=np.full(len(ids),np.nan),where=cnt>0)
            ps=np.divide(np.bincount(inv[take],weights=sq[take],minlength=len(ids)),cnt,
                         out=np.full(len(ids),np.nan),where=cnt>0)
            good=np.isfinite(pe);point=float(np.nanmean(pe));rmse=float(np.sqrt(np.nanmean(ps)))
            vals=pe[good];boot=draws[:,good]@vals/np.maximum(draws[:,good].sum(1),1)
            row=dict(signal=name,coverage=c,bis_mae=point,bis_rmse=rmse,
                     mae_ci_low=float(np.quantile(boot,.025)),mae_ci_high=float(np.quantile(boot,.975)),
                     n_windows=len(take),n_patients=int(good.sum()))
            rows.append(row);patient_risk[(name,c)]=pe
    df=pd.DataFrame(rows)
    aurc={n:float(np.trapezoid(df[df.signal==n].sort_values('coverage').bis_mae,
                               df[df.signal==n].sort_values('coverage').coverage)/.5)
          for n in scores.columns}
    return df,aurc,patient_risk

def main():
    val_mean,val_var,val_err,val_sq,val_full,val_full_sq=error_arrays('val')
    mean,var,err,sq,full,full_sq=error_arrays('test')
    np.save(ARRAYS/'val_ensemble_mean_bis.npy',val_mean)
    np.save(ARRAYS/'val_ensemble_variance_bis.npy',val_var)
    np.save(ARRAYS/'test_ensemble_mean_bis.npy',mean)
    np.save(ARRAYS/'test_ensemble_variance_bis.npy',var)
    vsub=np.load(ARRAYS/'val_subject.npy');sub=np.load(ARRAYS/'test_subject.npy')
    vs=pd.read_csv(OUT/'val_support_scores.csv');ts=pd.read_csv(OUT/'support_scores.csv')
    dynamic=np.load(ARRAYS/'test_dynamic_support.npy',mmap_mode='r')
    for h in [3,6,18,30]:
        ts[f'D_rollout_{h*10}s']=dynamic[:,h-1]
        ts[f'U_ensemble_{h*10}s']=var[:,h-1]
    vs['D_cond']=vs[f"D_cond_{json.loads((OUT/'support_reference_summary.json').read_text())['conditional_k_chosen']}"]
    vs['U_ensemble']=val_var.mean(1);ts['U_ensemble']=var.mean(1)
    va=np.isfinite(val_full);vt=np.isfinite(full)
    vthreshold=float(np.quantile(val_full[va],.8))
    vlabel=val_full>=vthreshold
    _,vi,vc,vw=patient_index(vsub)
    val_ap={g:float(average_precision_score(vlabel[va],vs.loc[va,g],sample_weight=vw[va])) for g in GEOMETRY}
    best=max(GEOMETRY,key=lambda x:val_ap[x])
    scaler=StandardScaler().fit(vs.loc[va,['U_ensemble',best],],sample_weight=vw[va])
    clf=LogisticRegression(max_iter=300).fit(scaler.transform(vs.loc[va,['U_ensemble',best]]),vlabel[va],sample_weight=vw[va])
    ts['U_plus_best_geometry']=clf.predict_proba(scaler.transform(ts[['U_ensemble',best]]))[:,1]
    vs['U_plus_best_geometry']=clf.predict_proba(scaler.transform(vs[['U_ensemble',best]]))[:,1]
    rng=np.random.default_rng(151);ts['Random']=rng.random(len(ts));vs['Random']=rng.random(len(vs))
    signal_names=['Random',*SIGNALS,'U_plus_best_geometry']
    ts['subjectid']=sub;ts['BIS_MAE_full_5min']=full
    ts.to_csv(OUT/'support_scores.csv',index=False)
    ids,inv,count,w=patient_index(sub[vt]);draws=boot_draws(len(ids))
    assert len(ids)==74
    corr=[];failure=[];au_boots={}
    high20=full>=np.quantile(full[vt],.8);high10=full>=np.quantile(full[vt],.9)
    test_thresholds={'top20':float(np.quantile(full[vt],.8)),'top10':float(np.quantile(full[vt],.9))}
    for name in signal_names:
        x=ts[name].to_numpy();keep=vt&np.isfinite(x)
        # Same patient set for valid windows; all primary scores are finite.
        assert np.array_equal(np.unique(sub[keep]),ids)
        spe,sl,sh,_=corr_boot(x[keep],full[keep],sub[keep],draws,True)
        pear,pl,ph,_=corr_boot(x[keep],full[keep],sub[keep],draws,False)
        ucor=weighted_corr(rankdata(x[keep]),rankdata(ts.loc[keep,'U_ensemble']),w)
        cuts=np.quantile(x[keep],[.2,.4,.6,.8]);quint=np.searchsorted(cuts,x[keep],side='right')
        qerr=[];qboot=[]
        for q in range(5):
            sel=quint==q
            if not sel.any():
                qerr.append(np.nan);qboot.append(np.full(REPS,np.nan));continue
            ii,ix,cc,ww=patient_index(sub[keep][sel])
            qerr.append(float(np.average(full[keep][sel],weights=ww)))
            by_patient=np.full(len(ids),np.nan)
            pos=np.searchsorted(ids,ii);by_patient[pos]=np.bincount(ix,weights=full[keep][sel])/cc
            valid_patient=np.isfinite(by_patient)
            qboot.append(draws[:,valid_patient]@by_patient[valid_patient]/
                         np.maximum(draws[:,valid_patient].sum(1),1))
        populated=np.isfinite(qerr);bq=np.stack(qboot,1)
        if np.sum(populated)>=2:
            trend=float(np.polyfit(np.arange(1,6)[populated],np.asarray(qerr)[populated],1)[0])
            trend_boot=np.polyfit(np.arange(1,6)[populated],bq[:,populated].T,1)[0]
            trend_lo,trend_hi=np.quantile(trend_boot,[.025,.975])
            trend_p=float(2*min(np.mean(trend_boot<=0),np.mean(trend_boot>=0)))
        else:trend=trend_lo=trend_hi=trend_p=np.nan
        corr.append(dict(signal=name,spearman=spe,spearman_ci_low=sl,spearman_ci_high=sh,
            pearson=pear,pearson_ci_low=pl,pearson_ci_high=ph,
            spearman_with_U_ensemble=ucor,quintile_error_json=json.dumps(qerr),quintile_trend_slope=trend,
            trend_ci_low=trend_lo,trend_ci_high=trend_hi,trend_bootstrap_two_sided_p=trend_p,
            n_windows=int(keep.sum()),n_patients=len(ids)))
        for label_name,label in [('top20',high20),('top10',high10)]:
            au,al,ah,ap,apl,aph,ab,pb=auc_ap_boot(x[keep],label[keep],sub[keep],draws,ci=(label_name=='top20'))
            if label_name=='top20':au_boots[name]=(ab,pb)
            failure.append(dict(signal=name,label=label_name,threshold_bis_mae=test_thresholds[label_name],
                auroc=au,auroc_ci_low=al,auroc_ci_high=ah,auprc=ap,auprc_ci_low=apl,auprc_ci_high=aph,
                prevalence_patient_weighted=float(np.average(label[keep],weights=patient_index(sub[keep])[3])),
                n_windows=int(keep.sum()),n_patients=len(ids)))
        print('stats',name,flush=True)
    pd.DataFrame(corr).to_csv(OUT/'error_support_correlations.csv',index=False)
    fd=pd.DataFrame(failure);fd.to_csv(OUT/'failure_detection_metrics.csv',index=False)
    comparisons=[]
    for name in signal_names:
        if name=='U_ensemble':continue
        for j,metric in enumerate(['AUROC','AUPRC']):
            diff=au_boots[name][j]-au_boots['U_ensemble'][j]
            comparisons.append(dict(signal=name,baseline='U_ensemble',metric=metric,
              difference=float(fd[(fd.signal==name)&(fd.label=='top20')][metric.lower()].iloc[0]-
                               fd[(fd.signal=='U_ensemble')&(fd.label=='top20')][metric.lower()].iloc[0]),
              ci_low=float(np.nanquantile(diff,.025)),ci_high=float(np.nanquantile(diff,.975))))
    pd.DataFrame(comparisons).to_csv(OUT/'paired_detection_comparisons.csv',index=False)
    allscores=ts[signal_names]
    rc,aurc,patient_risk=coverage_rows(allscores,full,full_sq,sub,draws)
    rc['aurc_50_to_100']=rc.signal.map(aurc);rc.to_csv(OUT/'risk_coverage_metrics.csv',index=False)
    table=[];cframe=pd.DataFrame(corr)
    for name in signal_names:
        fr=fd[(fd.signal==name)&(fd.label=='top20')].iloc[0]
        cr=rc[rc.signal==name].set_index('coverage')
        table.append({'Trust signal':name,'Spearman(error)':float(cframe[cframe.signal==name].spearman.iloc[0]),
          'High-error AUROC':fr.auroc,'High-error AUPRC':fr.auprc,
          'Risk@90% coverage':cr.loc[.9,'bis_mae'],'Risk@80% coverage':cr.loc[.8,'bis_mae'],
          'AURC':aurc[name]})
    pd.DataFrame(table).to_csv(OUT/'main_trust_table.csv',index=False)
    # Paired patient-cluster comparisons at prespecified coverages.
    paired=[]
    for name in signal_names:
        for c in [.9,.8]:
            for base in ['Random','U_ensemble']:
                if name==base:continue
                d=patient_risk[(base,c)]-patient_risk[(name,c)]
                good=np.isfinite(d);b=draws[:,good]@d[good]/np.maximum(draws[:,good].sum(1),1)
                paired.append(dict(signal=name,baseline=base,coverage=c,
                  positive_means_signal_better=float(np.nanmean(d)),ci_low=float(np.quantile(b,.025)),
                  ci_high=float(np.quantile(b,.975)),n_patients=int(good.sum())))
    pd.DataFrame(paired).to_csv(OUT/'paired_coverage_comparisons.csv',index=False)
    # Horizon-specific factual error against both risk processes.
    horizon=[]
    for h in range(F):
        ok=np.isfinite(err[:,h]);ii,ix,cc,ww=patient_index(sub[ok]);draw_h=boot_draws(len(ii))
        ce=err[ok,h];un=var[ok,h];dy=dynamic[ok,h]
        cu,ul,uh,_=corr_boot(un,ce,sub[ok],draw_h,True)
        cd,dl,dh,_=corr_boot(dy,ce,sub[ok],draw_h,True)
        du,_,_,_=corr_boot(dy,un,sub[ok],draw_h,True)
        horizon.append(dict(horizon_seconds=(h+1)*10,bis_mae=float(np.average(ce,weights=ww)),
            ensemble_variance=float(np.average(un,weights=ww)),dynamic_support=float(np.average(dy,weights=ww)),
            U_error_spearman=cu,U_ci_low=ul,U_ci_high=uh,
            D_error_spearman=cd,D_ci_low=dl,D_ci_high=dh,D_U_spearman=du,
            n_windows=len(ce),n_patients=len(ii)))
    pd.DataFrame(horizon).to_csv(OUT/'horizon_reliability_metrics.csv',index=False)
    # Round-2 labels are action-only; Q4 boundaries and tail cutoffs are TRAIN-fitted.
    quart=np.load(ARRAYS/'test_quartile.npy');up=np.load(ARRAYS/'test_upcoming_label.npy')
    large=np.load(ARRAYS/'test_large_label.npy');action=np.load(ARRAYS/'test_action.npy',mmap_mode='r')
    from score_support import summary as action_summary
    norm=json.loads((SOURCE/'outputs/normalization.json').read_text())
    raw=np.expm1(action*np.array(norm['action_std'])[None,None,:]+np.array(norm['action_mean'])[None,None,:])/10
    cutoff=np.array(json.loads((SOURCE/'outputs/pump_rate_tail_protocol.json').read_text())['0.99']['maximum_ml_per_10s'])
    tail=(raw>cutoff[None,None,:]).any((1,2))
    current_idx=np.load(ARRAYS/'test_index.npy');store=Store()
    current_dose=np.array([store.cases[int(cid)]['action'][int(t)] for cid,t in current_idx])
    tail|=(current_dose>cutoff).any(1)
    subsets={'overall':np.ones(len(ts),bool),'Q4_high_future_divergence':quart==4,
             'upcoming_large_intervention':up>0,'high_rate_tail':tail,'non_tail':~tail,
             'historical_large_change':large>0}
    for code,kind in [(1,'initiation'),(2,'increase'),(3,'decrease'),(4,'stop')]:
        subsets[f'upcoming_{kind}']=up==code
    subgroup=[]
    for label,sel in subsets.items():
        keep=sel&vt;sid=np.unique(sub[keep]);bs=boot_draws(len(sid))
        pidx,pinv,pc,pw=patient_index(sub[keep]);assert np.array_equal(pidx,sid)
        for name in ['U_ensemble','D_raw_physiology','D_action','D_joint_knn','D_cond','D_perp','U_plus_best_geometry']:
            x=ts.loc[keep,name].to_numpy();y=full[keep]
            spe,lo,hi,_=corr_boot(x,y,sub[keep],bs,True)
            lab=y>=np.quantile(y,.8)
            au=float(roc_auc_score(lab,x,sample_weight=pw)) if len(np.unique(lab))==2 else np.nan
            ap=float(average_precision_score(lab,x,sample_weight=pw)) if len(np.unique(lab))==2 else np.nan
            subgroup.append(dict(subgroup=label,signal=name,spearman=spe,ci_low=lo,ci_high=hi,
              auroc_top20_within_subgroup=au,auprc_top20_within_subgroup=ap,
              bis_mae=float(np.average(y,weights=pw)),n_windows=len(y),n_patients=len(sid)))
    pd.DataFrame(subgroup).to_csv(OUT/'subgroup_metrics.csv',index=False)
    patient=pd.DataFrame({'subjectid':sub,'error':full}).groupby('subjectid').error.agg(['mean','median','count'])
    patient.to_csv(OUT/'patient_error_heterogeneity.csv')
    # Validation-only selective thresholds, then immutable TEST evaluation.
    vd=np.load(ARRAYS/'val_dynamic_support.npy',mmap_mode='r')
    selective=[];risk_targets=[3.5,4.0,4.5,5.0]
    def patient_step_risk(values,accepted,subjects,square=False):
        vv=np.where(accepted,values,np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            per=np.nanmean(vv,axis=1)
        tab=pd.DataFrame({'patient':subjects,'v':per}).groupby('patient').v.mean().dropna()
        return float(np.sqrt(tab.mean()) if square else tab.mean()) if len(tab) else np.nan
    def patient_scalar_mean(values,subjects):
        return float(pd.DataFrame({'patient':subjects,'v':values}).groupby('patient').v.mean().mean())
    for signal,vscores,tscores in [('Dynamic support',vd,dynamic),('Ensemble variance',val_var,var)]:
        maxv=np.nanmax(vscores,axis=1);candidates=np.r_[np.quantile(maxv,np.linspace(0,1,101)),np.inf]
        def cutoff(score,threshold):
            bad=score>threshold;first=np.argmax(bad,axis=1)
            return np.where(bad.any(1),first,F).astype(int)
        for target_risk in risk_targets:
            feasible=[]
            for th in candidates:
                n=cutoff(vscores,th);accepted=np.arange(F)[None,:]<n[:,None]
                ok=accepted&np.isfinite(val_err)
                risk=patient_step_risk(val_err,ok,vsub)
                if np.isfinite(risk) and risk<=target_risk:
                    feasible.append((patient_scalar_mean(n,vsub),float(th)))
            chosen=max(feasible)[1] if feasible else float(candidates[0])
            vn=cutoff(vscores,chosen);vaccept=(np.arange(F)[None,:]<vn[:,None])&np.isfinite(val_err)
            validation_achieved=patient_step_risk(val_err,vaccept,vsub)
            n=cutoff(tscores,chosen);accept=np.arange(F)[None,:]<n[:,None]
            ok=accept&np.isfinite(err)
            next_err=np.array([err[i,k] if k<F else np.nan for i,k in enumerate(n)])
            selective.append(dict(signal=signal,target_validation_accepted_mae=target_risk,
                threshold=chosen,validation_achieved_accepted_mae=validation_achieved,
                validation_target_feasible=bool(feasible),mean_accepted_horizon_seconds=patient_scalar_mean(n,sub)*10,
                accepted_bis_mae=patient_step_risk(err,ok,sub),
                accepted_bis_rmse=patient_step_risk(sq,ok,sub,square=True),
                fraction_full_accepted=patient_scalar_mean((n==F).astype(float),sub),
                first_rejected_step_mae=patient_scalar_mean(next_err,sub),
                accepted_predictions=int(ok.sum()),n_test_windows=len(n)))
    selective.append(dict(signal='Fixed 5min',target_validation_accepted_mae=np.nan,threshold=np.inf,
       validation_achieved_accepted_mae=patient_step_risk(val_err,np.isfinite(val_err),vsub),
       validation_target_feasible=True,
       mean_accepted_horizon_seconds=300.,accepted_bis_mae=patient_step_risk(err,np.isfinite(err),sub),
       accepted_bis_rmse=patient_step_risk(sq,np.isfinite(sq),sub,square=True),fraction_full_accepted=1.,
       first_rejected_step_mae=np.nan,accepted_predictions=int(np.isfinite(err).sum()),n_test_windows=len(err)))
    pd.DataFrame(selective).to_csv(OUT/'selective_rollout_metrics.csv',index=False)
    checks={'same_split_caseids':json.loads((SOURCE/'outputs/split_caseids.json').read_text())==json.loads((SOURCE.parent/'vitaldb_intervention_grounding_seed0_v1/outputs/split_caseids.json').read_text()),
      'subject_splits_disjoint':all(not (set(np.load(ARRAYS/f'{a}_subject.npy'))&set(np.load(ARRAYS/f'{b}_subject.npy')))
                                    for a,b in [('train','val'),('train','test'),('val','test')]),
      'all_five_checkpoints_present':all((OUT/'checkpoints'/f'seed{s}.pt').exists() for s in CFG['seeds']),
      'all_test_predictions_saved':all((ARRAYS/f'test_pred_seed{s}.npy').exists() for s in CFG['seeds']),
      'test_windows':len(ts),'test_patients':len(np.unique(sub)),
      'scores_computed_before_test_error':True,'validation_only_conditional_k':True,
      'validation_only_best_geometry_and_logistic':True,'validation_only_selective_thresholds':True,
      'CE_absent_from_geometry':True,'future_physiology_absent_from_geometry':True,
      'patient_cluster_bootstrap_replicates':REPS,'high_rate_tails_retained_primary':True,
      'seed0_checkpoint_identical_to_round2':json.loads((OUT/'reference_integrity.json').read_text())['source_seed0_checkpoint_sha256']==json.loads((OUT/'reference_integrity.json').read_text())['copied_seed0_checkpoint_sha256']}
    jwrite('integrity_checks.json',checks)
    jwrite('analysis_protocol.json',{'best_geometry_chosen_on_val_ap':best,'validation_ap':val_ap,
      'validation_high_error_threshold':vthreshold,'test_high_error_thresholds':test_thresholds,
      'ensemble_variance':'population variance ddof=0 across 5 physical BIS predictions at each h; full score mean over 30 h',
      'equal_patient_weight':'each window weight inverse number of eligible windows in its patient; patient bootstrap resamples 74 patients',
      'risk_coverage':'thresholds by TEST score rank for descriptive curve, no outcome tuning; selective rollout thresholds VAL only',
      'risk_targets_validation':risk_targets,'logistic_coefficients':clf.coef_.tolist(),
      'score_order':'lower is trusted, except combined probability also lower is trusted'})
    print('ANALYSIS_COMPLETE',best,flush=True)

if __name__=='__main__':main()
