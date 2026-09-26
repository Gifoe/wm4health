"""Held-out forecasting, frozen probes, paired action audits and patient bootstrap."""
import json
import math
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors
from core import *

OUT=ROOT/'outputs'
HORIZONS=[3,6,18,30]

def bootstrap(values,subjects,reps=1000):
    """Equal-patient mean and cluster bootstrap, with all paired windows retained."""
    df=pd.DataFrame({'subject':subjects,'value':values}).dropna()
    v=df.groupby('subject').value.mean().to_numpy()
    if len(v)<2: return (float(np.nanmean(v)) if len(v) else np.nan,np.nan,np.nan,len(v))
    rng=np.random.default_rng(0)
    draws=v[rng.integers(0,len(v),(reps,len(v)))].mean(1)
    return float(v.mean()),float(np.quantile(draws,.025)),float(np.quantile(draws,.975)),len(v)

@torch.no_grad()
def infer(model,ds,store,shift=0,scale=1.,donors=None,return_latent=False):
    loader=DataLoader(ds,batch_size=512,num_workers=0,pin_memory=True)
    pred=[]; latent=[]
    for b in loader:
        if donors is not None:
            idx=b['index'].numpy(); replacement=b['action'].numpy().copy()
            for q,i in enumerate(idx):
                donor=donors[i]
                if donor<0: continue
                cid,t=ds.indices[donor]; c=store.cases[int(cid)]
                replacement[q]=np.concatenate([c['an'][t-179:t+1],c['action_mask'][t-179:t+1]],-1)
            b['action']=torch.from_numpy(replacement)
        b=to_device(b,'cuda')
        b=perturb_actions(b,store,shift=shift,scale=scale)
        p,z=model(b)
        pred.append(p.cpu().numpy())
        if return_latent: latent.append(z.cpu().numpy())
    return np.concatenate(pred),np.concatenate(latent) if return_latent else None

def get_targets(ds,store):
    n=len(ds); y=np.zeros((n,30,2),np.float32); mask=np.zeros_like(y,dtype=bool)
    for i,(cid,t) in enumerate(ds.indices):
        c=store.cases[int(cid)]
        y[i]=c['state'][t+1:t+31][:,[0,2]]
        mask[i]=c['state_fresh'][t+1:t+31][:,[0,2]]
    return y,mask

def physical(pred,store):
    return pred*store.norm['state_std'][[0,2]]+store.norm['state_mean'][[0,2]]

def errors(pred,y,mask):
    abs_error=np.where(mask,np.abs(pred-y),np.nan)
    sq_error=np.where(mask,(pred-y)**2,np.nan)
    return abs_error,sq_error

def forecast_rows(name,pred,y,mask,ref,subset='overall',keep=None):
    if keep is None: keep=np.ones(len(y),bool)
    ae,se=errors(pred,y,mask); rows=[]
    for j,target in enumerate(['BIS','MAP']):
        for h in HORIZONS+[0]:
            if h:
                e=ae[:,h-1,j]; s=se[:,h-1,j]
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore'); e=np.nanmean(ae[:,:,j],1); s=np.nanmean(se[:,:,j],1)
            e=e[keep]; s=s[keep]; sub=ref['subject'][keep]
            mean,lo,hi,npats=bootstrap(e,sub)
            mse,ml,mh,_=bootstrap(s,sub)
            rows.append({'model':name,'subset':subset,'target':target,'horizon_seconds':h*10,
                         'horizon_label':f'{h*10}s' if h else 'full_5min_trajectory',
                         'mae':mean,'mae_ci_low':lo,'mae_ci_high':hi,
                         'rmse':math.sqrt(mse) if mse>=0 else np.nan,
                         'rmse_ci_low':math.sqrt(ml) if ml>=0 else np.nan,
                         'rmse_ci_high':math.sqrt(mh) if mh>=0 else np.nan,
                         'pooled_mae':float(np.nanmean(e)) if len(e) else np.nan,
                         'n_windows':int(np.isfinite(e).sum()),'n_patients':npats})
    return rows

def drug_features(ds,store):
    """Uncompressed 180x2 action history; training-normalized, no physiology/demographics."""
    x=np.empty((len(ds),362),np.float32)
    for i,(cid,t) in enumerate(ds.indices):
        c=store.cases[int(cid)]
        # Linear in raw doses rather than log dose: per-drug train scale.
        a=c['action'][t-179:t+1]
        x[i,:360]=np.nan_to_num(a).reshape(-1)
        x[i,360:]=np.isnan(a).mean(0)
    return x

def fit_probe(x,y,subjects):
    """Ridge, train-only standardization, equal total weight per patient."""
    finite=np.isfinite(y); xx=x[finite].astype(np.float64); yy=y[finite].astype(np.float64)
    ss=subjects[finite]; uniq,cnt=np.unique(ss,return_counts=True)
    look=dict(zip(uniq,cnt)); w=np.array([1/look[s] for s in ss]); w*=len(w)/w.sum()
    xm=np.average(xx,axis=0,weights=w); ym=np.average(yy,weights=w)
    xc=xx-xm; sd=np.sqrt(np.average(xc**2,axis=0,weights=w)); sd[sd<1e-6]=1
    xc/=sd; yc=yy-ym
    gram=xc.T@(xc*w[:,None]); rhs=xc.T@(yc*w)
    coef=np.linalg.solve(gram+CFG['ridge_alpha']*np.eye(x.shape[1]),rhs)
    return {'coef':coef,'mean':xm,'std':sd,'intercept':ym}

def predict_probe(p,x): return ((x-p['mean'])/p['std'])@p['coef']+p['intercept']

def probe_metrics(name,ce,pred,ref,subset,keep):
    rows=[]
    for j,drug in enumerate(['PPF_CE','RFTN_CE']):
        k=keep&np.isfinite(ce[:,j])&np.isfinite(pred[:,j]); yy=ce[k,j]; pp=pred[k,j]; subjects=ref['subject'][k]
        if not len(yy): continue
        unique,cnt=np.unique(subjects,return_counts=True); lut=dict(zip(unique,cnt)); w=np.array([1/lut[s] for s in subjects]); w/=w.sum()
        ym=np.sum(w*yy); pm=np.sum(w*pp)
        r2=1-np.sum(w*(yy-pp)**2)/max(np.sum(w*(yy-ym)**2),1e-12)
        corr=np.sum(w*(yy-ym)*(pp-pm))/max(np.sqrt(np.sum(w*(yy-ym)**2)*np.sum(w*(pp-pm)**2)),1e-12)
        mae,lo,hi,npat=bootstrap(abs(yy-pp),subjects)
        # Weighted R2 and correlation are pooled with equal patient mass;
        # CIs resample patient sufficient statistics, retaining serial dependence.
        records=[]
        for s in unique:
            a=yy[subjects==s]; b=pp[subjects==s]
            records.append([np.mean(a),np.mean(b),np.mean(a*a),np.mean(b*b),np.mean(a*b),np.mean((a-b)**2)])
        records=np.array(records); rng=np.random.default_rng(0)
        draws=records[rng.integers(0,len(records),(1000,len(records)))].mean(1)
        var_y=draws[:,2]-draws[:,0]**2; var_p=draws[:,3]-draws[:,1]**2
        br=1-draws[:,5]/np.maximum(var_y,1e-12)
        bc=(draws[:,4]-draws[:,0]*draws[:,1])/np.sqrt(np.maximum(var_y*var_p,1e-12))
        rows.append({'representation':name,'drug':drug,'subset':subset,'r2':float(r2),
                     'r2_ci_low':np.quantile(br,.025),'r2_ci_high':np.quantile(br,.975),
                     'mae':mae,'mae_ci_low':lo,'mae_ci_high':hi,'pearson':float(corr),
                     'pearson_ci_low':np.quantile(bc,.025),'pearson_ci_high':np.quantile(bc,.975),
                     'n_windows':len(yy),'n_patients':npat,'probe_features':name})
    return rows

def match_donors(ds,store,ref):
    """Same test pool, different patient, calipered state match, different drug history."""
    complete=np.isfinite(ref['current']).all(1); ii=np.flatnonzero(complete)
    state=(ref['current']-store.norm['state_mean'])/store.norm['state_std']
    history=drug_features(ds,store)[:,:360]
    # Scale raw action differences by TRAIN positive administration SD.
    scales=[]
    for j in range(2):
        a=np.concatenate([store.cases[c]['action'][:,j] for c in store.splits['train']])
        scales.append(max(float(np.nanstd(a)),.001))
    scaled=history/np.tile(scales,180)
    donors=np.full(len(ds),-1,np.int32); rows=[]
    if len(ii)<2: return donors,pd.DataFrame(rows)
    nn=NearestNeighbors(n_neighbors=min(128,len(ii)),n_jobs=6).fit(state[ii])
    for start in range(0,len(ii),2048):
        ids=ii[start:start+2048]; dist,nei=nn.kneighbors(state[ids])
        for k,i in enumerate(ids):
            for d,p in zip(dist[k],nei[k]):
                q=ii[p]
                if ref['subject'][q]==ref['subject'][i]: continue
                if np.max(np.abs(state[q]-state[i]))>.5: continue
                action_rms=float(np.sqrt(np.mean((scaled[q]-scaled[i])**2)))
                if action_rms < .25: continue
                donors[i]=q
                rows.append({'window_index':int(i),'caseid':int(ref['case'][i]),'t':int(ref['t'][i]),
                             'donor_window_index':int(q),'donor_caseid':int(ref['case'][q]),
                             'donor_t':int(ref['t'][q]),'state_distance':float(d),
                             'bis_difference':float(ref['current'][q,0]-ref['current'][i,0]),
                             'hr_difference':float(ref['current'][q,1]-ref['current'][i,1]),
                             'map_difference':float(ref['current'][q,2]-ref['current'][i,2]),
                             'action_rms_train_sd':action_rms})
                break
    return donors,pd.DataFrame(rows)

def perturbation_size(ds,store,shift):
    out=np.zeros(len(ds),np.float32)
    ids=np.clip(np.arange(180)-int(shift/10),0,179)
    for i,(cid,t) in enumerate(ds.indices):
        a=store.cases[int(cid)]['action'][t-179:t+1]
        out[i]=np.nanmean(np.abs(a[ids]-a))
    return out

def main():
    seed_all(); store=Store(); ds=Windows(store,'test'); train=Windows(store,'train')
    ref=ds.reference(); trref=train.reference(); y,mask=get_targets(ds,store)
    event=ref['event']>0
    np.savez_compressed(OUT/'test_reference.npz',**ref,indices=ds.indices,y=y,mask=mask)
    pd.DataFrame(store.event_records).to_csv(OUT/'drug_change_events.csv',index=False)
    donors,matches=match_donors(ds,store,ref); matches.to_csv(OUT/'matched_shuffle_pairs.csv',index=False)
    np.save(OUT/'matched_shuffle_donor_indices.npy',donors)
    (OUT/'matched_shuffle_summary.json').write_text(json.dumps({'test_windows':len(ds),
        'matched_windows':int((donors>=0).sum()),'matched_fraction':float((donors>=0).mean()),
        'different_patient':True,'state_caliper_each_train_sd':.5,'minimum_action_history_rms_train_sd':.25,
        'state_absolute_difference_medians':matches[['bis_difference','hr_difference','map_difference']].abs().median().to_dict() if len(matches) else {}},indent=2))
    forecast=[]; timing=[]; ablation=[]; probes=[]; predictions={}; control_predictions={}
    subsets={'overall':np.ones(len(ds),bool),'transition':event}
    for kind,num in [('initiation',1),('increase',2),('decrease',3),('stop',4)]: subsets[kind]=ref['event']==num
    current=ref['current'][:,[0,2]]
    persistence=np.repeat(current[:,None,:],30,axis=1); predictions['Persistence']=persistence
    for sub,keep in subsets.items(): forecast+=forecast_rows('Persistence',persistence,y,mask,ref,sub,keep)
    for kind in ['Current physiology','Raw drug history','Physiology + demographics','Drug history + demographics']:
        print('PROBE_CONTROL '+kind,flush=True)
        if kind in ['Current physiology','Physiology + demographics']:
            # Availability flags prevent zero-imputation from meaning normal physiology.
            def xcurrent(rr):
                x=(rr['current']-store.norm['state_mean'])/store.norm['state_std']
                return np.c_[np.nan_to_num(x),np.isfinite(x).astype(float)].astype(np.float32)
            xtr=xcurrent(trref); xte=xcurrent(ref)
        else: xtr=drug_features(train,store); xte=drug_features(ds,store)
        if 'demographics' in kind:
            dtr=np.stack([store.cases[int(cid)]['dn'] for cid,_ in train.indices])
            dte=np.stack([store.cases[int(cid)]['dn'] for cid,_ in ds.indices])
            xtr=np.c_[xtr,dtr]; xte=np.c_[xte,dte]
        pp=np.full_like(ref['ce'],np.nan,dtype=float)
        for j in range(2):
            fitted=fit_probe(xtr,trref['ce'][:,j],trref['subject']); pp[:,j]=predict_probe(fitted,xte)
            np.savez(OUT/f'probe_{filename(kind)}_{j}.npz',**fitted)
        for sub,keep in subsets.items(): probes+=probe_metrics(kind,ref['ce'],pp,ref,sub,keep)
        control_predictions[kind]=pp
        np.save(OUT/f'ce_prediction_{filename(kind)}.npy',pp)
        del xtr,xte
    for name in MODEL_NAMES:
        print('AUDIT_MODEL '+name,flush=True)
        model=model_for(name).cuda()
        checkpoint=torch.load(OUT/'checkpoints'/f'{filename(name)}.pt',map_location='cuda',weights_only=False)
        model.load_state_dict(checkpoint['model']); model.eval()
        for p in model.parameters(): p.requires_grad_(False)
        norm,z=infer(model,ds,store,return_latent=True); pred=physical(norm,store); predictions[name]=pred
        np.savez_compressed(OUT/f'predictions_{filename(name)}.npz',prediction=pred,latent=z)
        for sub,keep in subsets.items(): forecast+=forecast_rows(name,pred,y,mask,ref,sub,keep)
        pd.DataFrame(forecast).to_csv(OUT/'forecast_metrics.csv',index=False)
        if name=='State-only': continue
        print('FROZEN_LATENTS_TRAIN '+name,flush=True)
        _,ztr=infer(model,train,store,return_latent=True)
        ce_pred=np.full_like(ref['ce'],np.nan,dtype=float)
        for j in range(2):
            fitted=fit_probe(ztr,trref['ce'][:,j],trref['subject']); ce_pred[:,j]=predict_probe(fitted,z)
            np.savez(OUT/f'probe_{filename(name)}_{j}.npz',**fitted)
        for sub,keep in subsets.items(): probes+=probe_metrics(name,ref['ce'],ce_pred,ref,sub,keep)
        np.save(OUT/f'ce_prediction_{filename(name)}.npy',ce_pred)
        del ztr
        pd.DataFrame(probes).to_csv(OUT/'ce_probe_metrics.csv',index=False)
        true_ae,true_se=errors(pred,y,mask); true_win=np.nanmean(true_ae[:,:,0],axis=1)
        for shift in CFG['time_shifts_seconds']:
            shifted=pred if shift==0 else physical(infer(model,ds,store,shift=shift)[0],store)
            dose_l1=perturbation_size(ds,store,shift)
            metrics=forecast_rows(name,shifted,y,mask,ref)
            sae,sse=errors(shifted,y,mask); shifted_win=np.nanmean(sae[:,:,0],1)
            for sub,keep in subsets.items():
                for row in forecast_rows(name,shifted,y,mask,ref,sub,keep):
                    if row['target']!='BIS': continue
                    h=row['horizon_seconds']//10
                    err=shifted_win if h==0 else sae[:,h-1,0]
                    base=true_win if h==0 else true_ae[:,h-1,0]
                    delta,lo,hi,_=bootstrap((err-base)[keep],ref['subject'][keep])
                    base_mae=bootstrap(base[keep],ref['subject'][keep])[0]
                    row.update({'shift_seconds':shift,'delta_mae':delta,'delta_ci_low':lo,'delta_ci_high':hi,
                                'normalized_degradation':delta/base_mae if base_mae>0 else np.nan,
                                'action_mean_abs_change_ml_per_10s':float(np.mean(dose_l1[keep])) if keep.any() else np.nan})
                    timing.append(row)
            print(f'TIMING {name} {shift}',flush=True)
        pd.DataFrame(timing).to_csv(OUT/'time_shift_metrics.csv',index=False)
        zero=physical(infer(model,ds,store,scale=0)[0],store)
        wrong=physical(infer(model,ds,store,donors=donors)[0],store)
        for label,changed,eligible in [('zero_action',zero,np.ones(len(ds),bool)),('matched_shuffle',wrong,donors>=0)]:
            ae,se=errors(changed,y,mask); e=np.nanmean(ae[:,:,0],1)
            for sub,keep in subsets.items():
                kk=keep&eligible
                for h in [0]+HORIZONS:
                    ee=e if h==0 else ae[:,h-1,0]; base=true_win if h==0 else true_ae[:,h-1,0]
                    delta,lo,hi,npat=bootstrap((ee-base)[kk],ref['subject'][kk])
                    baseline=bootstrap(base[kk],ref['subject'][kk])[0]
                    changed_error=bootstrap(ee[kk],ref['subject'][kk])[0]
                    ablation.append({'model':name,'intervention':label,'subset':sub,'horizon_seconds':h*10,
                                     'true_mae':baseline,'perturbed_mae':changed_error,'degradation':delta,
                                     'delta_ci_low':lo,'delta_ci_high':hi,'normalized_degradation':delta/baseline if baseline>0 else np.nan,
                                     'n_windows':int(kk.sum()),'n_patients':npat})
        pd.DataFrame(ablation).to_csv(OUT/'action_ablation_metrics.csv',index=False)
        np.savez_compressed(OUT/f'ablation_predictions_{filename(name)}.npz',zero=zero,wrong=wrong)
        # Representative patients fixed by sorted test-case quantiles, before errors.
        ids=sorted(store.splits['test']); selected=[ids[int(q*(len(ids)-1))] for q in [.15,.5,.85]]
        chosen=[]
        for cid in selected:
            rows=np.flatnonzero(ref['case']==cid)
            if len(rows):
                transitioned=rows[event[rows]]
                candidates=transitioned if len(transitioned) else rows
                chosen.append(int(candidates[len(candidates)//2]))
        small=Windows(store,'test',indices=ds.indices[chosen]); scale_preds=[]
        for alpha in CFG['dose_scales']:
            scale_preds.append(physical(infer(model,small,store,scale=alpha)[0],store))
        np.savez_compressed(OUT/f'dose_scaling_{filename(name)}.npz',prediction=np.stack(scale_preds),
                            alpha=CFG['dose_scales'],window_indices=chosen)
    fdf=pd.DataFrame(forecast); fdf.to_csv(OUT/'forecast_metrics.csv',index=False)
    fdf[fdf.subset!='overall'].to_csv(OUT/'transition_event_metrics.csv',index=False)
    pd.DataFrame(probes).to_csv(OUT/'ce_probe_metrics.csv',index=False)
    pd.DataFrame(probes).query("subset != 'overall'").to_csv(OUT/'transition_ce_probe_metrics.csv',index=False)
    pd.DataFrame(timing).query("subset != 'overall'").to_csv(OUT/'transition_time_shift_metrics.csv',index=False)
    improvements=[]
    for name in MODEL_NAMES:
        for against in ['Persistence','State-only']:
            if name==against: continue
            p=predictions[name]; baseline=predictions[against]
            for sub,keep in subsets.items():
                for h in [0]+HORIZONS:
                    a,_=errors(p,y,mask); b,_=errors(baseline,y,mask)
                    x=np.nanmean(a[:,:,0],1) if h==0 else a[:,h-1,0]
                    xx=np.nanmean(b[:,:,0],1) if h==0 else b[:,h-1,0]
                    diff,lo,hi,npats=bootstrap((xx-x)[keep],ref['subject'][keep])
                    base=bootstrap(xx[keep],ref['subject'][keep])[0]
                    improvements.append({'model':name,'reference_model':against,'subset':sub,'horizon_seconds':h*10,
                                         'mae_improvement':diff,'ci_low':lo,'ci_high':hi,'relative_improvement':diff/base,
                                         'n_patients':npats})
    pd.DataFrame(improvements).to_csv(OUT/'paired_forecast_comparisons.csv',index=False)
    print('EVALUATION_COMPLETE',flush=True)

if __name__=='__main__': main()
