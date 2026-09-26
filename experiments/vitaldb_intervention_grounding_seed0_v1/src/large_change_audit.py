"""Disclosed supplementary audit after tiny initial change thresholds were detected.

The original results are retained. Larger-event thresholds use TRAIN rates only:
max(original change threshold, 0.5 * IQR of positive train administration).
No refitting, model selection, probe changes, or new forecasting cases.
"""
import datetime,json
import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view
from core import *
from evaluate import forecast_rows,probe_metrics,bootstrap,infer,physical,errors,perturbation_size

OUT=ROOT/'outputs'

def integrate_results():
    mapping={'forecast_metrics':'large_change_forecast_metrics',
             'ce_probe_metrics':'large_change_ce_probe_metrics',
             'time_shift_metrics':'large_change_time_shift_metrics',
             'action_ablation_metrics':'large_change_action_ablation_metrics',
             'paired_forecast_comparisons':'large_change_paired_comparisons'}
    for target,source in mapping.items():
        original=pd.read_csv(OUT/(target+'.csv'))
        original=original[~original.subset.str.startswith('large_')]
        pd.concat([original,pd.read_csv(OUT/(source+'.csv'))],ignore_index=True).to_csv(OUT/(target+'.csv'),index=False)
    for source,target in [('forecast_metrics','transition_event_metrics'),('ce_probe_metrics','transition_ce_probe_metrics'),('time_shift_metrics','transition_time_shift_metrics')]:
        data=pd.read_csv(OUT/(source+'.csv'))
        data[data.subset!='overall'].to_csv(OUT/(target+'.csv'),index=False)

def main():
    seed_all(); store=Store(); ds=Windows(store,'test')
    with np.load(OUT/'test_reference.npz') as data: ref={k:data[k] for k in data.files}
    thresholds=[]
    for j in range(2):
        a=np.concatenate([store.cases[c]['action'][:,j] for c in store.splits['train']])
        pos=a[np.isfinite(a)&(a>0)]
        q=np.quantile(pos,[.1,.25,.5,.75,.9])
        thresholds.append({'original_change':store.thresholds[j]['change'],
                           'positive_rate_quantiles_10_25_50_75_90':q.tolist(),
                           'on':store.thresholds[j]['on'],
                           'large_change':max(store.thresholds[j]['change'],float(.5*(q[3]-q[1])))})
    tags={}; event_rows=[]
    for cid,c in store.cases.items():
        a=c['action']; n=len(a); med=np.full_like(a,np.nan)
        med[5:]=np.nanmedian(sliding_window_view(a,6,axis=0),axis=-1)
        pre=np.full_like(a,np.nan); pre[6:]=med[:-6]
        labels=np.zeros(n,np.int8)
        for j in range(2):
            on=thresholds[j]['on']; change=thresholds[j]['large_change']
            cs=np.r_[0,np.cumsum(np.isfinite(a[:,j]))]
            finite=np.zeros(n,bool); finite[11:]=(cs[12:]-cs[:-12])==12
            lab=np.zeros(n,np.int8)
            lab[finite&(pre[:,j]<=1e-6)&(med[:,j]>on)]=1
            lab[finite&(pre[:,j]>on)&(med[:,j]<=1e-6)]=4
            ongoing=finite&(pre[:,j]>1e-6)&(med[:,j]>1e-6)
            lab[ongoing&(med[:,j]-pre[:,j]>=change)]=2
            lab[ongoing&(pre[:,j]-med[:,j]>=change)]=3
            last=-100
            for t in np.flatnonzero(lab):
                if t-last<12: continue
                last=t; labels[t:min(n,t+13)]=lab[t]
                event_rows.append({'caseid':cid,'subjectid':int(c['subjectid']),'drug':j,'index':int(t),
                                   'kind':['none','initiation','increase','decrease','stop'][lab[t]],
                                   'pre':float(pre[t,j]),'post':float(med[t,j])})
        tags[cid]=labels
    kind=np.array([tags[int(cid)][t] for cid,t in ds.indices]); keep=kind>0
    protocol={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'status':'supplementary protocol added after inspecting initial Transformer timing results',
              'reason':'initial upper-quartile nonzero changes were about 0.35 mL/h and include micro-adjustments',
              'definition':'max(original change threshold, half IQR of positive TRAIN rates); zero crossings unchanged',
              'thresholds_ml_per_10s':thresholds,'n_test_windows':int(keep.sum()),
              'n_test_patients':int(np.unique(ref['subject'][keep]).size),
              'n_test_cases':int(np.unique(ref['case'][keep]).size),
              'kind_window_counts':{str(k):int((kind==k).sum()) for k in range(5)},
              'no_model_or_probe_refitting':True,'original_results_retained':True}
    (OUT/'large_change_protocol.json').write_text(json.dumps(protocol,indent=2))
    pd.DataFrame(event_rows).to_csv(OUT/'large_change_events.csv',index=False)
    np.save(OUT/'large_change_window_labels.npy',kind)
    print('LARGE_EVENT_PROTOCOL '+json.dumps(protocol),flush=True)
    forecast=[]; probes=[]; ablation=[]; timing=[]; comparisons=[]
    subjects=ref['subject']; y=ref['y']; mask=ref['mask']; donors=np.load(OUT/'matched_shuffle_donor_indices.npy')
    subsets={'large_transition':keep}
    for k,s in [(1,'large_initiation'),(2,'large_increase'),(3,'large_decrease'),(4,'large_stop')]: subsets[s]=kind==k
    preds={'Persistence':np.repeat(ref['current'][:,None,[0,2]],30,axis=1)}
    for name in MODEL_NAMES: preds[name]=np.load(OUT/f'predictions_{filename(name)}.npz')['prediction']
    for name,pred in preds.items():
        for sub,k in subsets.items(): forecast+=forecast_rows(name,pred,y,mask,ref,sub,k)
    reps=['Current physiology','Raw drug history','Physiology + demographics','Drug history + demographics']+MODEL_NAMES[1:]
    for name in reps:
        pred=np.load(OUT/f'ce_prediction_{filename(name)}.npy')
        for sub,k in subsets.items(): probes+=probe_metrics(name,ref['ce'],pred,ref,sub,k)
    subds=Windows(store,'test',indices=ds.indices[keep]); subref={k:v[keep] for k,v in ref.items()}; suby=y[keep]; submask=mask[keep]
    for name in MODEL_NAMES[1:]:
        trueae,_=errors(preds[name],y,mask); truewin=np.nanmean(trueae[:,:,0],1)
        saved=np.load(OUT/f'ablation_predictions_{filename(name)}.npz')
        for label,key,eligible in [('zero_action','zero',np.ones(len(ds),bool)),('matched_shuffle','wrong',donors>=0)]:
            ae,_=errors(saved[key],y,mask); changed=np.nanmean(ae[:,:,0],1)
            for sub,k in subsets.items():
                kk=k&eligible
                delta,lo,hi,npat=bootstrap((changed-truewin)[kk],subjects[kk]); base=bootstrap(truewin[kk],subjects[kk])[0]
                ablation.append({'model':name,'subset':sub,'intervention':label,'horizon_seconds':0,
                                 'true_mae':base,'perturbed_mae':base+delta,'degradation':delta,'delta_ci_low':lo,'delta_ci_high':hi,
                                 'normalized_degradation':delta/base,'n_windows':int(kk.sum()),'n_patients':npat})
        model=model_for(name).cuda(); ck=torch.load(OUT/'checkpoints'/f'{filename(name)}.pt',map_location='cuda',weights_only=False)
        model.load_state_dict(ck['model']); model.eval()
        for p in model.parameters(): p.requires_grad_(False)
        for shift in CFG['time_shifts_seconds']:
            shifted=preds[name][keep] if shift==0 else physical(infer(model,subds,store,shift=shift)[0],store)
            sae,_=errors(shifted,suby,submask); changed=np.nanmean(sae[:,:,0],1); basewin=truewin[keep]
            dose=perturbation_size(subds,store,shift)
            for row in forecast_rows(name,shifted,suby,submask,subref,'large_transition'):
                if row['target']!='BIS': continue
                h=row['horizon_seconds']//10
                err=changed if h==0 else sae[:,h-1,0]
                base=basewin if h==0 else trueae[keep,h-1,0]
                d,lo,hi,_=bootstrap(err-base,subref['subject']); baseline=bootstrap(base,subref['subject'])[0]
                row.update({'shift_seconds':shift,'delta_mae':d,'delta_ci_low':lo,'delta_ci_high':hi,
                            'normalized_degradation':d/baseline,'action_mean_abs_change_ml_per_10s':float(dose.mean())})
                timing.append(row)
            print('LARGE_TIMING',name,shift,flush=True)
        for other in ['Persistence','State-only']:
            bae,_=errors(preds[other],y,mask); base=np.nanmean(bae[:,:,0],1)
            d,lo,hi,npat=bootstrap((base-truewin)[keep],subjects[keep]); base_mae=bootstrap(base[keep],subjects[keep])[0]
            comparisons.append({'model':name,'reference_model':other,'subset':'large_transition','horizon_seconds':0,
                                'mae_improvement':d,'ci_low':lo,'ci_high':hi,'relative_improvement':d/base_mae,'n_patients':npat})
    for file,rows in [('large_change_forecast_metrics.csv',forecast),('large_change_ce_probe_metrics.csv',probes),
                      ('large_change_time_shift_metrics.csv',timing),('large_change_action_ablation_metrics.csv',ablation),
                      ('large_change_paired_comparisons.csv',comparisons)]:
        pd.DataFrame(rows).to_csv(OUT/file,index=False)
    integrate_results()
    print('LARGE_CHANGE_AUDIT_COMPLETE',flush=True)

if __name__=='__main__': main()
