"""Patient-cluster factual error, uncertainty calibration and support-shift audit."""
import hashlib,json,warnings
import numpy as np
import pandas as pd
import joblib
from scipy.stats import rankdata
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score,average_precision_score
from common import *

CONDITIONS=('F','L','Z','R');HORIZONS=(3,6,18,30)

def subjects(split):return np.asarray(arr(split,'subject'))

def patient_mean(v,sub,sel=None):
    if sel is None:sel=np.ones(len(v),bool)
    frame=pd.DataFrame({'sub':sub[sel],'v':np.asarray(v)[sel]})
    return float(frame.groupby('sub').v.mean().mean())

def patient_values(v,sub,sel):
    frame=pd.DataFrame({'sub':sub[sel],'v':np.asarray(v)[sel]})
    return frame.groupby('sub').v.mean()

def boot_mean(v,sub,sel,reps=1000,seed=0):
    p=patient_values(v,sub,sel).dropna().to_numpy();rng=np.random.default_rng(seed)
    draws=rng.integers(len(p),size=(reps,len(p)))
    b=p[draws].mean(1)
    return float(p.mean()),float(np.quantile(b,.025)),float(np.quantile(b,.975)),b

def boot_diff(a,b,sub,sel,reps=1000,seed=0):
    return boot_mean(np.asarray(a)-np.asarray(b),sub,sel,reps,seed)

def weighted_auc(score,label,sub,sel):
    s=np.asarray(score)[sel];y=np.asarray(label)[sel];subjects=sub[sel]
    _,inv,n=np.unique(subjects,return_inverse=True,return_counts=True)
    weight=1/n[inv]
    if len(np.unique(y))<2:return np.nan,np.nan
    return float(roc_auc_score(y,s,sample_weight=weight)),float(average_precision_score(y,s,sample_weight=weight))

def patient_weighted_quantile(v,sub,sel,q):
    x=np.asarray(v)[sel];subjects=sub[sel]
    _,inv,count=np.unique(subjects,return_inverse=True,return_counts=True)
    w=1/count[inv];order=np.argsort(x)
    return float(x[order][np.searchsorted(np.cumsum(w[order]),q*w.sum())])

def auc_boot_pair(a,b,label_a,label_b,sub,sel,reps=1000):
    ids=np.unique(sub[sel]);rng=np.random.default_rng(0);choices=rng.integers(len(ids),size=(reps,len(ids)))
    idx=np.flatnonzero(sel);local=sub[idx];by={p:idx[local==p] for p in ids}
    delta_auc=[];delta_ap=[]
    for draw in choices:
        mult=np.bincount(draw,minlength=len(ids));take=np.concatenate([by[ids[i]] for i in np.flatnonzero(mult)])
        w=np.concatenate([np.full(len(by[ids[i]]),mult[i]/len(by[ids[i]])) for i in np.flatnonzero(mult)])
        if len(np.unique(label_a[take]))<2 or len(np.unique(label_b[take]))<2:
            delta_auc.append(np.nan);delta_ap.append(np.nan);continue
        delta_auc.append(roc_auc_score(label_a[take],a[take],sample_weight=w)-roc_auc_score(label_b[take],b[take],sample_weight=w))
        delta_ap.append(average_precision_score(label_a[take],a[take],sample_weight=w)-average_precision_score(label_b[take],b[take],sample_weight=w))
    return np.nanquantile(delta_auc,[.025,.975]).tolist(),np.nanquantile(delta_ap,[.025,.975]).tolist()

def load_predictions(split,condition):
    p=np.stack([np.asarray(own(split,f'{condition}_pred_seed{seed}'))[:,:,0] for seed in CFG['seeds']])
    mean=p.mean(0);var=p.var(0,ddof=0)
    truth=np.asarray(arr(split,'target'))[:,:,0];mask=np.asarray(arr(split,'mask'))
    err=np.where(mask,np.abs(mean-truth),np.nan)
    sq=np.where(mask,(mean-truth)**2,np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        full=np.nanmean(err,axis=1);full_sq=np.nanmean(sq,axis=1)
    return dict(mean=mean,var=var,err=err,sq=sq,full=full,full_sq=full_sq,u=var.mean(1))

def calibrate(val,test,trainlike,vsub,tsub):
    # Separate monotone maps for trajectory and each prespecified point horizon.
    cal={};risk={};vpred={}
    _,inv,count=np.unique(vsub[trainlike],return_inverse=True,return_counts=True)
    w=np.empty(len(vsub));w[:]=np.nan;w[trainlike]=1/count[inv]
    for name,ve,vu,te,tu in [('full',val['full'],val['u'],test['full'],test['u'])]+[
       (str(h*10),val['err'][:,h-1],val['var'][:,h-1],test['err'][:,h-1],test['var'][:,h-1]) for h in HORIZONS]:
        ok=trainlike&np.isfinite(ve)&np.isfinite(vu)
        iso=IsotonicRegression(y_min=0,out_of_bounds='clip').fit(vu[ok],ve[ok],sample_weight=w[ok])
        cal[name]=iso;risk[name]=iso.predict(tu);vpred[name]=iso.predict(vu)
    return cal,risk,vpred

def main():
    seed_all(0);vsub=subjects('val');tsub=subjects('test')
    val_trainlike=np.load(ARRAYS/'val_trainlike.npy');target=np.load(ARRAYS/'test_target_cell.npy')
    s=np.load(ARRAYS/'test_state_cluster.npy');a=np.load(ARRAYS/'test_action_cluster.npy')
    chosen=[tuple(x) for x in json.loads((OUT/'selection_protocol.json').read_text())['selected_cells']]
    same_state=np.isin(s,[x[0] for x in chosen])&~target
    same_action=np.isin(a,[x[1] for x in chosen])&~target
    comparable=same_state|same_action
    comparable_same_patients=comparable&np.isin(tsub,np.unique(tsub[target]))
    norm=json.loads((SOURCE/'outputs/normalization.json').read_text())
    action_raw=np.expm1(np.asarray(arr('test','action'))*np.array(norm['action_std'])[None,None,:]+
                         np.array(norm['action_mean'])[None,None,:])/10
    cutoff=np.array(json.loads((SOURCE/'outputs/pump_rate_tail_protocol.json').read_text())['0.99']['maximum_ml_per_10s'])
    high_tail=(action_raw>cutoff).any((1,2))
    subsets={'target':target,'non_target':~target,'overall':np.ones(len(target),bool),
             'target_non_tail':target&~high_tail,'target_high_rate_tail':target&high_tail,
             'same_state_other_action':same_state,'same_action_other_state':same_action,
             'comparable_in_support':comparable,
             'comparable_same_patients':comparable_same_patients}
    result={};calibrators={};risk={};vrisk={};thresholds={}
    for c in CONDITIONS:
        val=load_predictions('val',c);test=load_predictions('test',c);result[c]=test
        calibrators[c],risk[c],vrisk[c]=calibrate(val,test,val_trainlike,vsub,tsub)
        joblib.dump(calibrators[c],OUT/f'calibration_{c}.joblib')
        threshold=patient_weighted_quantile(val['full'],vsub,val_trainlike,.8)
        severe=patient_weighted_quantile(val['full'],vsub,val_trainlike,.9)
        ucut=patient_weighted_quantile(val['u'],vsub,val_trainlike,.8)
        if c=='F':reference_error_thresholds=(threshold,severe)
        thresholds[c]=dict(high_error=reference_error_thresholds[0],
            severe_error=reference_error_thresholds[1],high_uncertainty=ucut,
            condition_specific_validation_error_p80=threshold,
            condition_specific_validation_error_p90=severe)
    jwrite('validation_thresholds.json',thresholds)
    metric_rows=[];horizon=[];cal_rows=[];cal_curve=[];quadrants=[];fail=[];cells=[]
    for c in CONDITIONS:
        t=result[c];r=risk[c]['full'];res=t['full']-r
        for subset,sel in subsets.items():
            mae,lo,hi,_=boot_mean(t['full'],tsub,sel)
            row=dict(condition=c,subset=subset,bis_mae_full=mae,mae_ci_low=lo,mae_ci_high=hi,
               bis_rmse_full=float(np.sqrt(patient_mean(t['full_sq'],tsub,sel))),
               ensemble_variance=patient_mean(t['u'],tsub,sel),predicted_risk=patient_mean(r,tsub,sel),
               calibration_residual=patient_mean(res,tsub,sel),windows=int(sel.sum()),patients=len(np.unique(tsub[sel])))
            metric_rows.append(row)
            cal_rows.append({k:row[k] for k in ('condition','subset','bis_mae_full','predicted_risk','calibration_residual','windows','patients')})
            if subset in ('target','non_target'):
                cuts=np.quantile(vrisk[c]['full'][val_trainlike],np.linspace(0,1,11)[1:-1])
                bins=np.searchsorted(cuts,r,side='right')
                for decile in range(10):
                    choose=sel&(bins==decile)
                    if not choose.any():continue
                    cal_curve.append(dict(condition=c,subset=subset,validation_risk_bin=decile+1,
                        predicted_risk=patient_mean(r,tsub,choose),actual_mae=patient_mean(t['full'],tsub,choose),
                        calibration_residual=patient_mean(res,tsub,choose),windows=int(choose.sum()),
                        patients=len(np.unique(tsub[choose]))))
            for frac,key in ((.8,'high_error'),(.9,'severe_error')):
                labels=t['full']>=thresholds[c][key]
                au,ap=weighted_auc(t['u'],labels,tsub,sel)
                fail.append(dict(condition=c,subset=subset,threshold=key,
                    validation_error_cutoff=thresholds[c][key],auroc=au,auprc=ap,
                    prevalence=patient_mean(labels.astype(float),tsub,sel),windows=int(sel.sum()),
                    patients=len(np.unique(tsub[sel]))))
            higherr=t['full']>=thresholds[c]['high_error']
            highu=t['u']>thresholds[c]['high_uncertainty']
            for u,e,label in [(False,False,'low_U_low_error'),(True,True,'high_U_high_error'),
                              (True,False,'high_U_low_error'),(False,True,'low_U_high_error')]:
                cat=(highu==u)&(higherr==e)
                q,ql,qh,_=boot_mean(cat.astype(float),tsub,sel)
                quadrants.append(dict(condition=c,subset=subset,quadrant=label,fraction=q,
                   ci_low=ql,ci_high=qh,windows=int(sel.sum()),patients=len(np.unique(tsub[sel]))))
            for h in HORIZONS:
                he=t['err'][:,h-1];hu=t['var'][:,h-1];hr=risk[c][str(h*10)]
                keep=sel&np.isfinite(he)
                horizon.append(dict(condition=c,subset=subset,horizon_seconds=h*10,
                    bis_mae=patient_mean(he,tsub,keep),predicted_risk=patient_mean(hr,tsub,keep),
                    bis_rmse=float(np.sqrt(patient_mean(t['sq'][:,h-1],tsub,keep))),
                    ensemble_variance=patient_mean(hu,tsub,keep),
                    calibration_residual=patient_mean(he-hr,tsub,keep),windows=int(keep.sum()),
                    patients=len(np.unique(tsub[keep]))))
        for sn,an in chosen:
            sel=(s==sn)&(a==an)
            e,elo,ehi,_=boot_mean(t['full'],tsub,sel)
            cells.append(dict(condition=c,state_cluster=sn,action_cluster=an,
              bis_mae_full=e,mae_ci_low=elo,mae_ci_high=ehi,
              ensemble_variance=patient_mean(t['u'],tsub,sel),
              predicted_risk=patient_mean(r,tsub,sel),
              calibration_residual=patient_mean(res,tsub,sel),
              confidently_wrong=patient_mean(((t['u']<=thresholds[c]['high_uncertainty'])&
                          (t['full']>=thresholds[c]['high_error'])).astype(float),tsub,sel),
              windows=int(sel.sum()),patients=len(np.unique(tsub[sel]))))
    pd.DataFrame(metric_rows).to_csv(OUT/'support_condition_metrics.csv',index=False)
    pd.DataFrame(cal_rows).to_csv(OUT/'uncertainty_calibration_metrics.csv',index=False)
    pd.DataFrame(cal_curve).to_csv(OUT/'uncertainty_calibration_curve.csv',index=False)
    pd.DataFrame(horizon).to_csv(OUT/'horizon_metrics.csv',index=False)
    pd.DataFrame(quadrants).to_csv(OUT/'confidence_error_quadrants.csv',index=False)
    pd.DataFrame(fail).to_csv(OUT/'failure_detection_metrics.csv',index=False)
    pd.DataFrame(cells).to_csv(OUT/'cell_level_metrics.csv',index=False)
    cell_pairs=[]
    for sn,an in chosen:
        sel=(s==sn)&(a==an)
        for label,c,d in [('low_minus_full','L','F'),('zero_minus_full','Z','F'),
                          ('zero_minus_random','Z','R')]:
            diff,lo,hi,_=boot_diff(result[c]['full'],result[d]['full'],tsub,sel)
            cell_pairs.append(dict(state_cluster=sn,action_cluster=an,comparison=label,
                difference=diff,ci_low=lo,ci_high=hi,patients=len(np.unique(tsub[sel])),windows=int(sel.sum())))
    pd.DataFrame(cell_pairs).to_csv(OUT/'cell_level_paired_comparisons.csv',index=False)
    comparisons=[]
    for label,c,d in [('low_minus_full','L','F'),('zero_minus_full','Z','F'),
                       ('random_minus_full','R','F'),('zero_minus_random','Z','R')]:
        for subset in ('target','target_non_tail','target_high_rate_tail','non_target','overall','comparable_in_support'):
            sel=subsets[subset]
            for metric,va,vb in [('error',result[c]['full'],result[d]['full']),
                                 ('uncertainty',result[c]['u'],result[d]['u']),
                                 ('calibration_residual',result[c]['full']-risk[c]['full'],result[d]['full']-risk[d]['full']),
                                 ('confidently_wrong',
                                  ((result[c]['u']<=thresholds[c]['high_uncertainty'])&(result[c]['full']>=thresholds[c]['high_error'])).astype(float),
                                  ((result[d]['u']<=thresholds[d]['high_uncertainty'])&(result[d]['full']>=thresholds[d]['high_error'])).astype(float))]:
                diff,lo,hi,_=boot_diff(va,vb,tsub,sel)
                comparisons.append(dict(comparison=label,subset=subset,metric=metric,difference=diff,
                                        ci_low=lo,ci_high=hi,bootstrap_replicates=1000,
                                        patients=len(np.unique(tsub[sel]))))
        # Support calibration gap is the difference of equal-patient group means.
        # The SAME resampled patient IDs drive both group means, retaining overlap.
        for condition in (c,d):
            v=result[condition]['full']-risk[condition]['full']
            x=patient_values(v,tsub,target);y=patient_values(v,tsub,comparable_same_patients)
            all_ids=np.union1d(x.index,y.index);xx=x.reindex(all_ids).to_numpy();yy=y.reindex(all_ids).to_numpy()
            rng=np.random.default_rng(0)
            draws=np.array([np.bincount(rng.integers(len(all_ids),size=len(all_ids)),minlength=len(all_ids))
                            for _ in range(1000)],dtype='float32')
            bx=(draws@np.nan_to_num(xx))/np.maximum(draws@np.isfinite(xx).astype(float),1)
            by=(draws@np.nan_to_num(yy))/np.maximum(draws@np.isfinite(yy).astype(float),1)
            bb=bx-by
            comparisons.append(dict(comparison=condition,subset='target_minus_comparable',metric='SCG',
              difference=float(np.nanmean(xx)-np.nanmean(yy)),ci_low=float(np.quantile(bb,.025)),
              ci_high=float(np.quantile(bb,.975)),bootstrap_replicates=1000,
              patients=len(all_ids),target_patients=int(np.isfinite(xx).sum()),
              comparable_patients=int(np.isfinite(yy).sum())))
        # Paired detection differences on identical target windows. Labels use
        # the common Full-Support validation-derived absolute-error thresholds.
        la=result[c]['full']>=thresholds[c]['high_error'];lb=result[d]['full']>=thresholds[d]['high_error']
        ac,pc=weighted_auc(result[c]['u'],la,tsub,target)
        ad,pd_=weighted_auc(result[d]['u'],lb,tsub,target)
        auc_ci,ap_ci=auc_boot_pair(result[c]['u'],result[d]['u'],la,lb,tsub,target)
        for name,value,ci in [('AUROC',ac-ad,auc_ci),('AUPRC',pc-pd_,ap_ci)]:
            comparisons.append(dict(comparison=label,subset='target',metric=name,difference=value,
                ci_low=ci[0],ci_high=ci[1],bootstrap_replicates=1000,
                patients=len(np.unique(tsub[target]))))
    for label,c,d in [('zero_minus_full','Z','F'),('zero_minus_random','Z','R'),('low_minus_full','L','F')]:
        residual_c=result[c]['full']-risk[c]['full']
        residual_d=result[d]['full']-risk[d]['full']
        shift=residual_c-residual_d
        q_target=patient_values(shift,tsub,target)
        q_control=patient_values(shift,tsub,comparable_same_patients)
        ids=q_target.index.intersection(q_control.index)
        per=q_target.loc[ids].to_numpy()-q_control.loc[ids].to_numpy()
        rng=np.random.default_rng(0)
        boot=per[rng.integers(len(per),size=(1000,len(per)))].mean(1)
        comparisons.append(dict(comparison=label,subset='target_minus_comparable',metric='SCG',
            difference=float(per.mean()),ci_low=float(np.quantile(boot,.025)),
            ci_high=float(np.quantile(boot,.975)),bootstrap_replicates=1000,patients=len(ids)))
    pd.DataFrame(comparisons).to_csv(OUT/'paired_comparisons.csv',index=False)
    # Export compact, strictly aligned per-window diagnostics for plotting.
    test_idx=np.asarray(arr('test','index'))
    per=pd.DataFrame({'caseid':test_idx[:,0],'anchor_t':test_idx[:,1],'subjectid':tsub,
       'state_cluster':s,'action_cluster':a,'target':target})
    for c in CONDITIONS:
        per[f'error_{c}']=result[c]['full'];per[f'uncertainty_{c}']=result[c]['u']
        per[f'predicted_risk_{c}']=risk[c]['full'];per[f'prediction_endpoint_{c}']=result[c]['mean'][:,-1]
    per.to_csv(OUT/'window_diagnostics.csv',index=False)
    jwrite('evaluation_summary.json',{'target_windows':int(target.sum()),'target_patients':len(np.unique(tsub[target])),
      'same_test_windows_all_conditions':True,'calibration_fit_split':'VAL_trainlike only',
      'risk_calibrator':'isotonic regression, per condition, equal-patient weighted',
      'error_and_uncertainty':'full BIS trajectory MAE; population variance across five physical predictions, averaged over 30 steps'})
    print('EVALUATION_COMPLETE',flush=True)

if __name__=='__main__':main()
