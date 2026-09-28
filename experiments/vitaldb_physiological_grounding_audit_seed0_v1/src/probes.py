"""TRAIN Ridge probes, VAL-only alpha selection, patient-weighted TEST evaluation."""
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from common import *

SPLITS=['train','val','test'];VARS=['BIS','MAP','HR'];CE=['PPF_CE','RFTN_CE']

def arrays(name):return {s:load(s,name) for s in SPLITS}

def feature(kind,hidx=None,exclude=None):
    result={}
    for s in SPLITS:
        z=np.asarray(load(s,'z0'))
        if kind=='z_t':v=z
        elif kind=='demographics':v=np.asarray(load(s,'raw_history'))[:,-4:]
        elif kind=='drug_history':v=np.asarray(load(s,'drug_history'))
        elif kind=='raw_other_current':
            q=np.asarray(load(s,'raw_history'));drop={exclude+3*k for k in range(5)}
            v=q[:,[i for i in range(19) if i not in drop]]
        elif kind=='raw_history':v=np.asarray(load(s,'raw_history'))
        elif kind=='action':v=np.asarray(load(s,'future_action_summary'))[:,hidx]
        elif kind=='z_t+action':v=np.concatenate([z,np.asarray(load(s,'future_action_summary'))[:,hidx]],1)
        elif kind=='raw_history+action':v=np.concatenate([np.asarray(load(s,'raw_history')),
                                         np.asarray(load(s,'future_action_summary'))[:,hidx]],1)
        elif kind=='dz_pred':v=np.asarray(load(s,'rollout'))[:,HORIZONS[hidx]-1]-z
        elif kind=='dz_true':v=np.asarray(load(s,'ztrue'))[:,hidx]-z
        elif kind=='random_latent':
            rng=np.random.default_rng({'train':101,'val':102,'test':103}[s]);v=rng.normal(size=z.shape).astype(np.float32)
        elif kind=='random_difference':
            rng=np.random.default_rng({'train':201,'val':202,'test':203}[s]);v=rng.normal(size=z.shape).astype(np.float32)
        else:raise ValueError(kind)
        result[s]=np.nan_to_num(v,nan=0,posinf=0,neginf=0).astype(np.float32)
    return result

def target(var,hidx=None,kind='change'):
    y={};valid={}
    for s in SPLITS:
        cur=np.asarray(load(s,'current'));future=np.asarray(load(s,'target'))
        cm=np.asarray(load(s,'current_fresh'));fm=np.asarray(load(s,'fresh'))
        if kind=='current':
            j=VARS.index(var);y[s]=cur[:,j].astype(np.float32)
            valid[s]=cm[:,j]&np.isfinite(y[s])
        elif kind=='change':
            j=VARS.index(var);y[s]=(future[:,hidx,j]-cur[:,j]).astype(np.float32)
            valid[s]=cm[:,j]&fm[:,hidx,j]&np.isfinite(y[s])
        elif kind=='ce_current':
            j=CE.index(var);y[s]=np.asarray(load(s,'ce'))[:,0,j].astype(np.float32)
            valid[s]=np.isfinite(y[s])
        elif kind=='ce_change':
            j=CE.index(var);ce=np.asarray(load(s,'ce'))
            y[s]=(ce[:,hidx+1,j]-ce[:,0,j]).astype(np.float32)
            valid[s]=np.isfinite(y[s])
        else:raise ValueError(kind)
    return y,valid

def fit_probe(features,y,valid,metric_type,representation,var,hidx=None):
    tr=valid['train']&np.isfinite(features['train']).all(1)
    va=valid['val']&np.isfinite(features['val']).all(1)
    te=valid['test']&np.isfinite(features['test']).all(1)
    if min(tr.sum(),va.sum(),te.sum())<100:return None,None
    scaler=StandardScaler().fit(features['train'][tr])
    xtrain=scaler.transform(features['train'][tr]);xval=scaler.transform(features['val'][va]);xtest=scaler.transform(features['test'])
    score=-np.inf;selected=None;alpha=None
    for a in CFG['ridge_alpha_grid']:
        model=Ridge(alpha=a,solver='cholesky').fit(xtrain,y['train'][tr])
        pv=model.predict(xval)
        # Evaluate only the held-out valid rows, preserving patient weighting.
        r=weighted_r2(y['val'][va],pv,np.asarray(load('val','subject'))[va],np.ones(va.sum(),bool))
        if np.isfinite(r) and r>score:score=r;selected=model;alpha=a
    if selected is None:return None,None
    pred=selected.predict(xtest).astype(np.float32)
    test_subject=np.asarray(load('test','subject'))
    result={'audit':metric_type,'representation':representation,'target':var,
            'horizon_seconds':HORIZONS[hidx]*10 if hidx is not None else 0,
            'alpha':alpha,'val_r2':score,'n_train_windows':int(tr.sum()),'n_val_windows':int(va.sum()),
            **weighted_metrics(y['test'],pred,test_subject,te)}
    if metric_type in ('transition','current'):
        result['r2_ci_low'],result['r2_ci_high']=patient_bootstrap_stat(y['test'],pred,test_subject,te,'r2',CFG['bootstrap_replicates'])
    return result,pred

def paired_transition(y,valid,pred,truth,hidx,var):
    subject=np.asarray(load('test','subject'));m=valid['test']&np.isfinite(pred)&np.isfinite(truth)
    ids=np.unique(subject[m]);parts=[]
    for s in ids:
        sub=m&(subject==s);v=y['test'][sub].astype(np.float64)
        pp=pred[sub].astype(np.float64);tt=truth[sub].astype(np.float64)
        parts.append([v.mean(),np.mean(v*v),np.mean((v-pp)**2),np.mean((v-tt)**2),
                      np.mean(np.abs(v-pp)),np.mean(np.abs(v-tt))])
    q=np.asarray(parts);rng=np.random.default_rng(0)
    d=q[rng.integers(len(ids),size=(CFG['bootstrap_replicates'],len(ids)))].mean(1)
    den=d[:,1]-d[:,0]**2
    r2diff=(d[:,2]-d[:,3])/den
    mae_diff=d[:,4]-d[:,5]
    point=weighted_r2(y['test'],truth,subject,m)-weighted_r2(y['test'],pred,subject,m)
    return {'comparison':'factual_minus_predicted','target':var,'horizon_seconds':HORIZONS[hidx]*10,
            'r2_difference':point,'r2_ci_low':float(np.quantile(r2diff,.025)),
            'r2_ci_high':float(np.quantile(r2diff,.975)),
            'mae_difference_pred_minus_factual':float(q[:,4].mean()-q[:,5].mean()),
            'mae_ci_low':float(np.quantile(mae_diff,.025)),'mae_ci_high':float(np.quantile(mae_diff,.975)),
            'patients':len(ids),'windows':int(m.sum())}

def main():
    OUT.mkdir(exist_ok=True);ARRAYS.mkdir(exist_ok=True)
    # Train-only physiological change scale for combined response magnitude.
    scales=[]
    for var in VARS:
        y,valid=target(var,3,'change');scales.append(float(np.nanstd(y['train'][valid['train']])))
    write_json('response_scale_protocol.json',{'train_300s_change_std':dict(zip(VARS,scales)),
              'magnitude':'Euclidean norm of BIS/MAP/HR changes divided by TRAIN 300s-change SD'})
    rows=[];current=[];future=[];transition=[];ce=[];paired=[];pred_cache={}
    for var in VARS:
        y,valid=target(var,kind='current');j=VARS.index(var)
        for rep in ['z_t','demographics','drug_history','raw_other_current','random_latent']:
            result,pred=fit_probe(feature(rep,exclude=j),y,valid,'current',rep,var)
            if result:current.append(result)
        print('CURRENT',var,flush=True)
    for k,h in enumerate(HORIZONS):
        for var in VARS:
            y,valid=target(var,k,'change')
            for rep in ['z_t','action','z_t+action','raw_history+action']:
                result,pred=fit_probe(feature(rep,k),y,valid,'future',rep,var,k)
                if result:future.append(result)
            for rep in ['dz_pred','dz_true','random_difference','raw_history+action']:
                result,pred=fit_probe(feature(rep,k),y,valid,'transition',rep,var,k)
                if result:
                    transition.append(result)
                    if rep in ('dz_pred','dz_true'):
                        pred_cache[(k,var,rep)]=pred
                        np.save(ARRAYS/f'probe_{rep}_{var}_{h}.npy',pred)
            if (k,var,'dz_pred') in pred_cache and (k,var,'dz_true') in pred_cache:
                paired.append(paired_transition(y,valid,pred_cache[(k,var,'dz_pred')],
                                                pred_cache[(k,var,'dz_true')],k,var))
            print('CHANGE',h,var,flush=True)
    for var in CE:
        y,valid=target(var,kind='ce_current')
        result,_=fit_probe(feature('z_t'),y,valid,'ce_reference','z_t',var)
        if result:ce.append(result)
        for k,h in enumerate(HORIZONS):
            y,valid=target(var,k,'ce_change')
            for rep in ['dz_pred','dz_true']:
                result,_=fit_probe(feature(rep,k),y,valid,'ce_reference',rep,var,k)
                if result:ce.append(result)
    pd.DataFrame(current).to_csv(OUT/'current_state_probe_metrics.csv',index=False)
    pd.DataFrame(future).to_csv(OUT/'future_change_probe_metrics.csv',index=False)
    pd.DataFrame(transition).to_csv(OUT/'transition_probe_metrics.csv',index=False)
    pd.DataFrame(ce).to_csv(OUT/'ce_reference_probe_metrics.csv',index=False)
    pd.DataFrame(paired).to_csv(OUT/'paired_comparisons.csv',index=False)
    print('RIDGE_PROBES_COMPLETE',flush=True)

if __name__=='__main__':main()
