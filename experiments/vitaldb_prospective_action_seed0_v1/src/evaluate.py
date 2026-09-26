"""Prospective schedule comparisons, fixed matched donors and patient-cluster uncertainty."""
import json,math,warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors
from core import *

OUT=ROOT/'outputs'
HORIZONS=[3,6,18,30,0]
KINDS={1:'initiation',2:'increase',3:'decrease',4:'stop'}

def bootstrap(values,subjects,reps=1000):
    df=pd.DataFrame({'subject':subjects,'v':values}).dropna()
    v=df.groupby('subject').v.mean().to_numpy()
    if not len(v): return np.nan,np.nan,np.nan,0
    if len(v)==1: return float(v[0]),np.nan,np.nan,1
    rng=np.random.default_rng(0)
    draws=v[rng.integers(len(v),size=(reps,len(v)))].mean(1)
    return float(v.mean()),float(np.quantile(draws,.025)),float(np.quantile(draws,.975)),len(v)

def physical(pred,store):
    return pred*store.norm['state_std'][[0,2]]+store.norm['state_mean'][[0,2]]

def abs_error(pred,target,mask):
    return np.where(mask,np.abs(pred-target),np.nan)

def scalar_error(error,h,j=0):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return np.nanmean(error[:,:,j],1) if h==0 else error[:,h-1,j]

def score_rows(model,condition,error,subjects,subsets):
    rows=[]
    for subset,keep in subsets.items():
        for j,target in enumerate(['BIS','MAP']):
            for h in HORIZONS:
                e=scalar_error(error,h,j);v,lo,hi,n=bootstrap(e[keep],subjects[keep])
                rows.append({'model':model,'condition':condition,'subset':subset,'target':target,
                             'horizon_seconds':h*10,'horizon_label':f'{h*10}s' if h else 'full_5min',
                             'mae':v,'ci_low':lo,'ci_high':hi,'n_windows':int(np.isfinite(e[keep]).sum()),
                             'n_patients':n})
    return rows

def paired_rows(errors,subjects,subsets,matched):
    rows=[]; true=errors['true']
    for subset,keep in subsets.items():
        for cond,label in [('hold','FAV'),('wrong','MFAV'),('zero','ZFAD')]:
            mask=keep & (matched if cond=='wrong' else True)
            for h in HORIZONS:
                e=scalar_error(errors[cond],h)-scalar_error(true,h)
                v,lo,hi,n=bootstrap(e[mask],subjects[mask])
                rows.append({'comparison':label,'condition':cond,'subset':subset,'horizon_seconds':h*10,
                             'difference_mae':v,'ci_low':lo,'ci_high':hi,
                             'n_windows':int(np.isfinite(e[mask]).sum()),'n_patients':n})
    return rows

def action_scales(store):
    return np.array([max(np.nanstd(np.concatenate([store.cases[c]['action'][:,j]
                                      for c in store.splits['train']])),.001) for j in range(2)],np.float32)

def future_divergence(future,last,scale):
    return np.sqrt(np.mean(((future-last[:,None,:])/scale)**2,axis=(1,2)))

def train_divergence(store,scale):
    values=[]
    for cid in store.splits['train']:
        c=store.cases[cid]; a=c['action']
        for t in c['anchors']:
            f=a[t+1:t+F+1]
            if np.isfinite(f).all() and np.isfinite(a[t]).all():
                values.append(float(np.sqrt(np.mean(((f-a[t])/scale)**2))))
    return np.asarray(values,np.float32)

def divergence_bins(train_d,test_d):
    boundaries=np.quantile(train_d,[.25,.5,.75]).astype(float)
    # Ties at zero are assigned to the lowest bin. Empty tied bins stay explicit.
    bins=np.searchsorted(boundaries,test_d,side='left')+1
    return bins,boundaries

def match_donors(ref,store,scale):
    current=ref['current']; complete=np.isfinite(current).all(1); ii=np.flatnonzero(complete)
    state=(current-store.norm['state_mean'])/store.norm['state_std']
    future=ref['future_action']/scale[None,None,:]
    donors=np.full(len(current),-1,np.int32); rows=[]
    nn=NearestNeighbors(n_neighbors=min(CFG['match_neighbors'],len(ii)),n_jobs=6).fit(state[ii])
    for start in range(0,len(ii),2048):
        ids=ii[start:start+2048];dist,near=nn.kneighbors(state[ids])
        for r,i in enumerate(ids):
            for dd,p in zip(dist[r],near[r]):
                q=int(ii[p])
                if ref['subject'][i]==ref['subject'][q]: continue
                if np.max(np.abs(state[i]-state[q]))>CFG['match_state_caliper_train_sd']: continue
                divergence=float(np.sqrt(np.mean((future[i]-future[q])**2)))
                if divergence<CFG['match_future_action_rms_train_sd']: continue
                donors[i]=q
                rows.append({'window_index':int(i),'caseid':int(ref['case'][i]),'anchor_t':int(ref['t'][i]),
                             'subjectid':int(ref['subject'][i]),'donor_index':q,
                             'donor_caseid':int(ref['case'][q]),'donor_subjectid':int(ref['subject'][q]),
                             'state_distance':float(dd),'max_state_difference_train_sd':float(np.max(np.abs(state[i]-state[q]))),
                             'future_action_rms_train_sd':divergence})
                break
    return donors,pd.DataFrame(rows)

@torch.no_grad()
def infer(model,ds,store,policy,donors=None,return_latent=False):
    model.eval(); device=next(model.parameters()).device
    loader=DataLoader(ds,batch_size=512,num_workers=0,pin_memory=True)
    predictions=[]; latents=[]
    for b in loader:
        if donors is not None:
            idx=b['index'].numpy(); f=b['future_action'].numpy()
            for q,i in enumerate(idx):
                donor=donors[i]
                if donor>=0:
                    cid,t=ds.indices[donor];c=store.cases[int(cid)]
                    f[q]=np.concatenate([c['an'][t+1:t+F+1],c['am'][t+1:t+F+1]],-1)
            b['future_action']=torch.from_numpy(f)
        b=to_device(b,device)
        p,_,z=model(b,policy,return_steps=return_latent)
        predictions.append(physical(p.cpu().numpy(),store))
        if return_latent: latents.append(z.cpu().numpy())
    return np.concatenate(predictions),np.concatenate(latents) if return_latent else None

def round1_large_labels(ds,store):
    original=[]
    for cid in store.splits['test']:
        original.extend((cid,int(t)) for t in store.cases[cid]['anchors'])
    saved=np.load(SOURCE/'outputs/large_change_window_labels.npy')
    assert len(original)==len(saved)==83992
    lookup=dict(zip(original,saved))
    return np.array([lookup[tuple(x)] for x in ds.indices],np.int8)

def upcoming_large_labels(ds):
    """Action-only future confirmations; >=60s after anchor to avoid pre-anchor change onset."""
    table=pd.read_csv(SOURCE/'outputs/large_change_events.csv')
    records={}
    for cid,g in table.groupby('caseid'):
        g=g.sort_values('index')
        records[int(cid)]=(g['index'].to_numpy(int),g['kind'].map({v:k for k,v in KINDS.items()}).to_numpy(int))
    result=np.zeros(len(ds),np.int8)
    for i,(cid,t) in enumerate(ds.indices):
        if int(cid) not in records:continue
        times,kinds=records[int(cid)];q=np.searchsorted(times,int(t)+6)
        if q<len(times) and times[q]<=t+F:result[i]=kinds[q]
    return result

def main():
    seed_all();store=Store();ds=Windows(store,'test');ref=ds.reference()
    labels=round1_large_labels(ds,store);upcoming=upcoming_large_labels(ds);subjects=ref['subject']
    scales=action_scales(store)
    last=np.array([store.cases[int(cid)]['action'][t] for cid,t in ds.indices],np.float32)
    divergence=future_divergence(ref['future_action'],last,scales)
    train_d=train_divergence(store,scales)
    bins,edges=divergence_bins(train_d,divergence)
    assert np.isfinite(edges).all() and np.isfinite(divergence).all()
    (OUT/'divergence_protocol.json').write_text(json.dumps({'train_quantile_boundaries':edges.tolist(),
       'train_windows_with_defined_current_and_future_action':int(len(train_d)),
       'scale_train_raw_action_sd':scales.tolist(),'zero_fraction_test':float((divergence==0).mean()),
       'bin_counts':{f'Q{i}':int((bins==i).sum()) for i in range(1,5)},
       'definition':'RMS over 30x2 physical actions, divided by train action SD; zero ties assigned lower bin'},indent=2))
    donors,pairs=match_donors(ref,store,scales)
    pairs.to_csv(OUT/'matched_future_action_pairs.csv',index=False)
    assert (ref['subject'][donors[donors>=0]]!=ref['subject'][np.flatnonzero(donors>=0)]).all()
    matched=donors>=0
    subsets={'overall':np.ones(len(ds),bool),'large_transition':labels>0}
    for code,kind in KINDS.items():subsets[kind]=labels==code
    subsets['upcoming_large_transition']=upcoming>0
    for code,kind in KINDS.items():subsets['upcoming_'+kind]=upcoming==code
    for q in range(1,5):subsets[f'Q{q}']=bins==q
    # A future change confirmed after t provides a more direct prospective-event subset.
    future_large=np.zeros(len(ds),bool)
    for i,(cid,t) in enumerate(ds.indices):
        # Future divergence measures the schedule itself; no future physiological labels.
        future_large[i]=divergence[i]>=max(edges[-1],.25)
    subsets['high_future_divergence']=future_large
    target=np.array([store.cases[int(cid)]['state'][t+1:t+F+1][:,[0,2]] for cid,t in ds.indices],np.float32)
    mask=np.array([store.cases[int(cid)]['state_fresh'][t+1:t+F+1][:,[0,2]] for cid,t in ds.indices],bool)
    all_errors={};forecast=[]
    for name,policy,condition in [('Historical RSSM','hold','hold'),('Prospective RSSM','true','true'),
                                   ('Prospective RSSM','hold','hold'),('Prospective RSSM','zero','zero'),
                                   ('Prospective RSSM','true','wrong')]:
        model=RSSM().cuda();model.set_action_stats(store)
        checkpoint=torch.load(OUT/'checkpoints'/('hold.pt' if name=='Historical RSSM' else 'true.pt'),map_location='cuda',weights_only=False)
        model.load_state_dict(checkpoint['model']);model.eval()
        pred,_=infer(model,ds,store,policy,donors=donors if condition=='wrong' else None)
        err=abs_error(pred,target,mask)
        key='historical' if name=='Historical RSSM' else condition
        all_errors[key]=err
        keep=matched if condition=='wrong' else np.ones(len(ds),bool)
        forecast+=score_rows(name,condition,err,subjects,{k:v&keep for k,v in subsets.items()})
        np.savez_compressed(OUT/f'prediction_{key}.npz',prediction=pred)
        print('INFER',name,condition,flush=True)
    pd.DataFrame(forecast).to_csv(OUT/'forecast_metrics.csv',index=False)
    paired=paired_rows(all_errors,subjects,subsets,matched)
    pd.DataFrame(paired).to_csv(OUT/'paired_comparisons.csv',index=False)
    pd.DataFrame(paired).to_csv(OUT/'prospective_action_metrics.csv',index=False)
    pd.DataFrame([r for r in paired if r['subset'].startswith('Q')]).to_csv(OUT/'future_action_divergence_metrics.csv',index=False)
    pd.DataFrame([r for r in paired if r['subset'] in ['large_transition',*KINDS.values(),
       'upcoming_large_transition',*[f'upcoming_{k}' for k in KINDS.values()],'high_future_divergence']]).to_csv(OUT/'large_transition_metrics.csv',index=False)
    np.savez_compressed(OUT/'test_reference.npz',case=ref['case'],t=ref['t'],subject=subjects,
                        target=target,mask=mask,large_label=labels,upcoming_label=upcoming,divergence=divergence,quartile=bins,
                        future_action=ref['future_action'],donor=donors)
    summary=json.loads((OUT/'data_summary.json').read_text());summary['test_matched_windows']=int(matched.sum())
    summary['test_patients_with_complete_future_action']=int(np.unique(subjects).size)
    summary['test_patients_in_split']=len(store.subject_splits['test'])
    summary['test_large_transition_round1_windows']=int((labels>0).sum())
    summary['test_high_future_divergence_windows']=int(future_large.sum())
    summary['test_upcoming_large_transition_windows']=int((upcoming>0).sum())
    (OUT/'data_summary.json').write_text(json.dumps(summary,indent=2))
    print('EVALUATION_COMPLETE',flush=True)

if __name__=='__main__':main()
