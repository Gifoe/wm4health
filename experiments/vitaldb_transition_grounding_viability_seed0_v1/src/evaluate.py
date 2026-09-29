"""Frozen-model extraction and matched Round-6 probing/geometry protocol."""
import json,sys,hashlib,warnings
from pathlib import Path
import numpy as np,pandas as pd,torch,yaml
from torch.utils.data import DataLoader
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1];CFG=yaml.safe_load((ROOT/'config.yaml').read_text())
SOURCE=(ROOT/CFG['source_experiment']).resolve();AUDIT=(ROOT/CFG['audit_experiment']).resolve()
sys.path.insert(0,str(SOURCE/'src'))
from core import Store,Windows,RSSM,to_device,seed_all
sys.path.insert(0,str(AUDIT/'src'))
import geometry as g
from common import weighted_metrics,weighted_r2,patient_weights

OUT=ROOT/'outputs';ARR=OUT/'arrays';ARR.mkdir(parents=True,exist_ok=True)
H=[3,6,18,30];VARS=['BIS','MAP','HR'];CE=['PPF_CE','RFTN_CE']
def old(split,key):return np.load(AUDIT/'outputs/arrays'/f'{split}_{key}.npy',mmap_mode='r')
def feat(model,split,key):
    if model=='Baseline':
        if key=='dz':return np.asarray(old(split,'rollout'))[:,np.asarray(H)-1]-np.asarray(old(split,'z0'))[:,None,:]
        return np.asarray(old(split,key))
    return np.load(ARR/f'{model}_{split}_{key}.npy',mmap_mode='r')

def target(split,var,k=None):
    if var in VARS:
        j=VARS.index(var);c=np.asarray(old(split,'current'))[:,j];cm=np.asarray(old(split,'current_fresh'))[:,j]
        if k is None:return c,cm&np.isfinite(c)
        f=np.asarray(old(split,'target'))[:,k,j];fm=np.asarray(old(split,'fresh'))[:,k,j]
        return f-c,cm&fm&np.isfinite(c)&np.isfinite(f)
    j=CE.index(var);c=np.asarray(old(split,'ce'))[:,0,j]
    if k is None:return c,np.isfinite(c)
    f=np.asarray(old(split,'ce'))[:,k+1,j]
    return f-c,np.isfinite(c)&np.isfinite(f)

def by_patient(y,p,subject,valid,metric):
    m=valid&np.isfinite(y)&np.isfinite(p);ids=np.unique(subject[m]);q=[]
    for s in ids:
        a=y[m&(subject==s)].astype(float);b=p[m&(subject==s)].astype(float)
        q.append([a.mean(),np.mean(a*a),np.mean((a-b)**2),np.mean(np.abs(a-b)),np.mean((a-b)**2)])
    q=np.asarray(q)
    if not len(q):return np.nan,np.nan,np.nan,0
    z=q.mean(0);value=(1-z[2]/(z[1]-z[0]**2)) if metric=='r2' else z[3]
    if len(q)<2:return value,np.nan,np.nan,len(q)
    rng=np.random.default_rng(0);draw=q[rng.integers(len(q),size=(1000,len(q)))].mean(1)
    v=1-draw[:,2]/(draw[:,1]-draw[:,0]**2) if metric=='r2' else draw[:,3]
    return float(value),*map(float,np.nanquantile(v,[.025,.975])),len(q)

def fit_ridge(model,var,k=None,mode='transition'):
    xs={s:np.nan_to_num(np.asarray(feat(model,s,'z0' if k is None else 'dz'))[:,k] if k is not None else np.asarray(feat(model,s,'z0'))) for s in ('train','val','test')}
    ys={s:target(s,var,k)[0] for s in xs};ms={s:target(s,var,k)[1] for s in xs}
    scl=StandardScaler().fit(xs['train'][ms['train']]);tr=scl.transform(xs['train'][ms['train']]);va=scl.transform(xs['val'][ms['val']]);te=scl.transform(xs['test'])
    best=-np.inf;selected=None;alpha=None
    for a in (.1,10.,1000.):
        probe=Ridge(alpha=a,solver='cholesky').fit(tr,ys['train'][ms['train']])
        score=weighted_r2(ys['val'][ms['val']],probe.predict(va),np.asarray(old('val','subject'))[ms['val']],np.ones(ms['val'].sum(),bool))
        if score>best:best=score;selected=probe;alpha=a
    pred=selected.predict(te).astype(np.float32)
    sub=np.asarray(old('test','subject'));met=weighted_metrics(ys['test'],pred,sub,ms['test'])
    return dict(model=model,target=var,horizon_seconds=0 if k is None else H[k]*10,alpha=alpha,val_r2=best,**met),pred

@torch.no_grad()
def extract(model_name,model,store,split):
    ds=Windows(store,split);n=len(ds);device=next(model.parameters()).device
    z0=np.lib.format.open_memmap(ARR/f'{model_name}_{split}_z0.npy','w+',dtype='float32',shape=(n,64))
    dz=np.lib.format.open_memmap(ARR/f'{model_name}_{split}_dz.npy','w+',dtype='float32',shape=(n,4,64))
    pred=np.lib.format.open_memmap(ARR/f'{model_name}_{split}_prediction.npy','w+',dtype='float32',shape=(n,30,2)) if split=='test' else None
    at=0
    for b in DataLoader(ds,batch_size=512,num_workers=0,pin_memory=True):
        b=to_device(b,device);p,z,r=model(b,'true',True);m=len(z)
        zz=z.cpu().numpy();z0[at:at+m]=zz;dz[at:at+m]=r[:,np.asarray(H)-1].cpu().numpy()-zz[:,None,:]
        if pred is not None:pred[at:at+m]=p.cpu().numpy()*store.norm['state_std'][[0,2]]+store.norm['state_mean'][[0,2]]
        at+=m
    z0.flush();dz.flush()
    if pred is not None:pred.flush()
    print('EXTRACT',model_name,split,n,flush=True)

@torch.no_grad()
def corrupt(model,store,policy,donors=None):
    ds=Windows(store,'test');device=next(model.parameters()).device;arr=[]
    for b in DataLoader(ds,batch_size=512,num_workers=0,pin_memory=True):
        if donors is not None:
            ix=b['index'].numpy();a=b['future_action'].numpy()
            for j,i in enumerate(ix):
                q=int(donors[i])
                if q>=0:
                    cid,t=ds.indices[q];c=store.cases[int(cid)]
                    a[j]=np.concatenate([c['an'][t+1:t+31],c['am'][t+1:t+31]],1)
            b['future_action']=torch.from_numpy(a)
        p,_,_=model(to_device(b,device),policy)
        arr.append(p.cpu().numpy()*store.norm['state_std'][[0,2]]+store.norm['state_mean'][[0,2]])
    return np.concatenate(arr)

def forecast_metric(p,ref,model):
    subject=ref['subject'];y=ref['target'];mask=ref['mask'].astype(bool)
    rows=[]
    for j,var in enumerate(['BIS','MAP']):
        e=np.where(mask[:,:,j],p[:,:,j]-y[:,:,j],np.nan)
        for h in H+[0]:
            x=e[:,h-1] if h else e
            valid=np.isfinite(x).any(1) if not h else np.isfinite(x)
            ids=np.unique(subject[valid]);per=[]
            for s in ids:
                v=x[valid&(subject==s)];per.append([np.nanmean(np.abs(v)),np.nanmean(v*v)])
            per=np.asarray(per)
            rows.append(dict(model=model,target=var,horizon_seconds=h*10,mae=float(per[:,0].mean()),rmse=float(np.sqrt(per[:,1].mean())),patients=len(ids),windows=int(valid.sum())))
    return rows

def subgroup_masks(ref,store):
    up=ref['upcoming_label'];tail=json.loads((SOURCE/'outputs/pump_rate_tail_protocol.json').read_text())['0.99']['maximum_ml_per_10s']
    act=ref['future_action'];cur=np.asarray([store.cases[int(cid)]['action'][int(t)] for cid,t in zip(ref['case'],ref['t'])])
    high=(act>np.asarray(tail)[None,None,:]).any((1,2))|(cur>np.asarray(tail)[None,:]).any(1)
    return {'overall':np.ones(len(up),bool),'stable_action_Q1':ref['quartile']==1,'Q4':ref['quartile']==4,
            'upcoming_large_intervention':up>0,'initiation':up==1,'increase':up==2,'decrease':up==3,
            'stop':up==4,'high_rate_tail':high,'non_tail':~high}

def subset_mae(p,ref,subset,h,j=0):
    e=np.where(ref['mask'][:,h-1,j]&subset,np.abs(p[:,h-1,j]-ref['target'][:,h-1,j]),np.nan)
    sub=ref['subject'];q=[np.nanmean(e[sub==s]) for s in np.unique(sub[np.isfinite(e)])]
    return float(np.nanmean(q)),len(q),int(np.isfinite(e).sum())

def paired_probe(preds,model_a,model_b,var,k):
    y,m=target('test',var,k);sub=np.asarray(old('test','subject'));a=preds[(model_a,var,k)];b=preds[(model_b,var,k)]
    valid=m&np.isfinite(a)&np.isfinite(b);ids=np.unique(sub[valid]);parts=[]
    for s in ids:
        q=valid&(sub==s);yy=y[q].astype(float);aa=a[q].astype(float);bb=b[q].astype(float)
        parts.append([yy.mean(),np.mean(yy*yy),np.mean((yy-aa)**2),np.mean((yy-bb)**2)])
    parts=np.asarray(parts);rng=np.random.default_rng(0);d=parts[rng.integers(len(parts),size=(1000,len(parts)))].mean(1)
    diff=(d[:,3]-d[:,2])/(d[:,1]-d[:,0]**2)
    return float(np.nanmean(diff)),*map(float,np.nanquantile(diff,[.025,.975])),len(ids)

def main():
    seed_all(0);store=Store();device='cuda' if torch.cuda.is_available() else 'cpu'
    selected=json.loads((OUT/'selected_lambda.json').read_text())
    models={};paths={'Baseline':SOURCE/'outputs/checkpoints/true.pt',
                     'State-Grounded':ROOT/'checkpoints/state_selected.pt',
                     'Transition-Grounded':ROOT/'checkpoints/transition_selected.pt'}
    for name,path in paths.items():
        ck=torch.load(path,map_location=device,weights_only=False);m=RSSM().to(device);m.set_action_stats(store);m.load_state_dict(ck['model']);m.eval();models[name]=m
    ref=np.load(SOURCE/'outputs/test_reference.npz');orig=np.load(SOURCE/'outputs/prediction_true.npz')['prediction']
    cached=np.asarray(old('test','prediction'))
    assert np.array_equal(orig,cached),'Baseline Round-6/2 mismatch'
    (OUT/'data_summary.json').write_text(json.dumps({'cases':495,'patients':493,'case_splits':{k:len(v) for k,v in store.splits.items()},
        'subject_splits':{k:len(v) for k,v in store.subject_splits.items()},'eligible_windows':{s:len(old(s,'subject')) for s in ('train','val','test')},
        'test_patients':int(np.unique(ref['subject']).size),'baseline_reproduced_exactly':True,'CE_training_use':False},indent=2))
    preds={'Baseline':orig}
    for name in ('State-Grounded','Transition-Grounded'):
        for split in ('train','val','test'):extract(name,models[name],store,split)
        preds[name]=np.asarray(feat(name,'test','prediction'))
    rows=[]
    for name,p in preds.items():rows+=forecast_metric(p,ref,name)
    pd.DataFrame(rows).to_csv(OUT/'forecast_metrics.csv',index=False)
    probe_preds={};cur=[];transition=[];ce=[]
    for name in models:
        if name=='Baseline':
            for filename,dest,representation in [('current_state_probe_metrics.csv',cur,'z_t'),
                                                  ('transition_probe_metrics.csv',transition,'dz_pred'),
                                                  ('ce_reference_probe_metrics.csv',ce,'dz_pred')]:
                original=pd.read_csv(AUDIT/'outputs'/filename)
                dest.extend(original[original.representation==representation].assign(model=name).to_dict('records'))
            for var in VARS:
                for k,h in enumerate(H):
                    probe_preds[(name,var,k)]=np.load(AUDIT/'outputs/arrays'/f'probe_dz_pred_{var}_{h}.npy',mmap_mode='r')
            continue
        for var in VARS:
            r,p=fit_ridge(name,var);cur.append(r)
            for k in range(4):
                r,p=fit_ridge(name,var,k);transition.append(r);probe_preds[(name,var,k)]=p
                np.save(ARR/f'probe_{name}_{var}_{H[k]}.npy',p)
        for var in CE:
            for k in range(4):
                r,p=fit_ridge(name,var,k);ce.append(r)
        print('PROBES',name,flush=True)
    pd.DataFrame(cur).to_csv(OUT/'current_state_probe_metrics.csv',index=False)
    pd.DataFrame(transition).to_csv(OUT/'transition_probe_metrics.csv',index=False)
    pd.DataFrame(ce).to_csv(OUT/'ce_reference_metrics.csv',index=False)
    # Same TRAIN thresholds, per-patient sample caps, and paired RNG as Round 6.
    protocol=json.loads((AUDIT/'outputs/response_group_protocol.json').read_text())
    scales=np.asarray(list(json.loads((AUDIT/'outputs/response_scale_protocol.json').read_text())['train_300s_change_std'].values()),np.float32)
    sub=np.asarray(old('test','subject'));masks=subgroup_masks(ref,store);directions=[];ordering=[];group_rows=[];patient_rows=[]
    g.CFG=yaml.safe_load((AUDIT/'config.yaml').read_text())
    for name in models:
        vt=np.asarray(feat(name,'train','dz'));vtest=np.asarray(feat(name,'test','dz'))
        for k,h in enumerate(H):
            y=np.asarray(old('test','target'))[:,k]-np.asarray(old('test','current'))
            valid=np.asarray(old('test','fresh'))[:,k]&np.asarray(old('test','current_fresh'))&np.isfinite(y)
            zscale=np.maximum(np.std(vt[:,k],axis=0),1e-3)
            for group,keep in masks.items():
                if group!='overall' and h not in (18,30):continue
                for j,var in enumerate(('BIS','MAP')):
                    cut=np.asarray(protocol['train_quintile_thresholds'][f'{var}_{h*10}'])
                    out=g.direction_metrics(vtest[:,k],y[:,j],valid[:,j],sub,cut,keep,group,name,h,var,
                                            30 if group=='overall' else 12,patient_rows)
                    directions+=out
                order,_=g.ordering_metrics(vtest[:,k],y,valid,sub,keep,group,name,h,scales,zscale,
                                           60 if group=='overall' else 20,patient_rows)
                if order:ordering.append(order)
                if h in (18,30):
                    for j,var in enumerate(VARS):
                        py=probe_preds[(name,var,k)];met=weighted_metrics(y[:,j],py,sub,valid[:,j]&keep)
                        fmae,npat,nwin=subset_mae(preds[name],ref,keep,h)
                        group_rows.append(dict(model=name,subset=group,horizon_seconds=h*10,target=var,
                                               probe_r2=met['r2'],probe_mae=met['mae'],bis_forecast_mae=fmae,
                                               patients=npat,windows=nwin))
            print('GEOMETRY',name,h,flush=True)
    pd.DataFrame(directions).to_csv(OUT/'transition_direction_metrics.csv',index=False)
    pd.DataFrame(ordering).to_csv(OUT/'response_ordering_metrics.csv',index=False)
    pd.DataFrame(group_rows).to_csv(OUT/'subgroup_metrics.csv',index=False)
    pd.DataFrame(patient_rows).to_csv(OUT/'geometry_patient_summary.csv',index=False)
    previous=pd.read_csv(AUDIT/'outputs/response_ordering_metrics.csv')
    now=pd.DataFrame(ordering)
    previous_trip=previous.query('subset=="overall" and representation=="dz_pred" and horizon_seconds==300').iloc[0].triplet_accuracy
    now_trip=now.query('subset=="overall" and representation=="Baseline" and horizon_seconds==300').iloc[0].triplet_accuracy
    assert abs(previous_trip-now_trip)<1e-6,(previous_trip,now_trip)
    gain=[];fm=pd.DataFrame(rows)
    for name in ('State-Grounded','Transition-Grounded'):
        for var in ('BIS','MAP'):
            b30=fm.query('model=="Baseline" and target==@var and horizon_seconds==30').iloc[0].mae
            b300=fm.query('model=="Baseline" and target==@var and horizon_seconds==300').iloc[0].mae
            n30=fm.query('model==@name and target==@var and horizon_seconds==30').iloc[0].mae
            n300=fm.query('model==@name and target==@var and horizon_seconds==300').iloc[0].mae
            for h in H+[0]:
                seconds=h*10
                base=fm.query('model=="Baseline" and target==@var and horizon_seconds==@seconds').iloc[0]
                got=fm.query('model==@name and target==@var and horizon_seconds==@seconds').iloc[0]
                gain.append(dict(model=name,target=var,horizon_seconds=h*10,gain_mae=base.mae-got.mae,
                                 baseline_mae=base.mae,model_mae=got.mae,
                                 baseline_growth_30_to_300=b300-b30,model_growth_30_to_300=n300-n30))
    pd.DataFrame(gain).to_csv(OUT/'horizon_gain_metrics.csv',index=False)
    action=[]
    for name,model in models.items():
        for condition in ('true','hold','wrong'):
            pred=preds[name] if condition=='true' else corrupt(model,store,'hold' if condition=='hold' else 'true',ref['donor'] if condition=='wrong' else None)
            if name=='Baseline' and condition!='true':
                prior=np.load(SOURCE/'outputs'/f'prediction_{condition}.npz')['prediction']
                assert np.max(np.abs(pred-prior))<1e-5,f'Baseline {condition} mismatch'
            for group in ('Q4','upcoming_large_intervention'):
                # All conditions use exactly the same matched-donor windows.
                keep=masks[group]&(ref['donor']>=0)
                mae,npat,nwin=subset_mae(pred,ref,keep,30)
                action.append(dict(model=name,condition=condition,subset=group,horizon_seconds=300,mae=mae,patients=npat,windows=nwin))
            print('ACTION',name,condition,flush=True)
    pd.DataFrame(action).to_csv(OUT/'action_sensitivity_metrics.csv',index=False)
    paired=[]
    for a,b in [('Transition-Grounded','Baseline'),('Transition-Grounded','State-Grounded')]:
        for var in VARS:
            d,lo,hi,n=paired_probe(probe_preds,a,b,var,3)
            paired.append(dict(comparison=f'{a} minus {b}',metric=f'300s_delta_{var}_r2',difference=d,ci_low=lo,ci_high=hi,patients=n))
        for var in ('BIS','MAP'):
            j=('BIS','MAP').index(var);y=ref['target'][:,29,j];m=ref['mask'][:,29,j]
            sub=ref['subject'];ids=np.unique(sub[m]);q=[]
            for s in ids:
                ix=m&(sub==s);q.append(np.mean(np.abs(preds[a][ix,29,j]-y[ix]))-np.mean(np.abs(preds[b][ix,29,j]-y[ix])))
            q=np.asarray(q);rng=np.random.default_rng(0);draw=q[rng.integers(len(q),size=(1000,len(q)))].mean(1)
            paired.append(dict(comparison=f'{a} minus {b}',metric=f'300s_{var}_mae',difference=float(q.mean()),
                               ci_low=float(np.quantile(draw,.025)),ci_high=float(np.quantile(draw,.975)),patients=len(q)))
        gp=pd.DataFrame(patient_rows)
        for metric in ('triplet_accuracy','BIS_same_minus_opposite_cosine'):
            w=gp.query('subset=="overall" and horizon_seconds==300 and metric==@metric').pivot_table(index='subject',columns='representation',values='value')
            q=(w[a]-w[b]).dropna().to_numpy();rng=np.random.default_rng(0);draw=q[rng.integers(len(q),size=(1000,len(q)))].mean(1)
            paired.append(dict(comparison=f'{a} minus {b}',metric=metric,difference=float(q.mean()),ci_low=float(np.quantile(draw,.025)),
                               ci_high=float(np.quantile(draw,.975)),patients=len(q)))
    pd.DataFrame(paired).to_csv(OUT/'paired_comparisons.csv',index=False)
    print('EVALUATION_COMPLETE',flush=True)
if __name__=='__main__':main()
