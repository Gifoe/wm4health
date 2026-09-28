"""Patient-balanced transition-direction and response-order geometry audits."""
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from common import *

VARS=['BIS','MAP','HR']

def delta_phys(split,k):
    c=np.asarray(load(split,'current'));f=np.asarray(load(split,'target'))[:,k]
    cm=np.asarray(load(split,'current_fresh'));fm=np.asarray(load(split,'fresh'))[:,k]
    y=f-c;valid=cm&fm&np.isfinite(y)
    return y,valid

def dz(split,k,rep):
    z=np.asarray(load(split,'z0'))
    return (np.asarray(load(split,'rollout'))[:,HORIZONS[k]-1]-z if rep=='dz_pred'
            else np.asarray(load(split,'ztrue'))[:,k]-z).astype(np.float32)

def groups():
    ref=np.load(SOURCE/'outputs/test_reference.npz')
    n=len(ref['subject']);up=ref['upcoming_label']
    return {'overall':np.ones(n,bool),'stable_action_Q1':ref['quartile']==1,
            'Q4':ref['quartile']==4,'upcoming_large_intervention':up>0,
            'initiation':up==1,'increase':up==2,'decrease':up==3,'stop':up==4}

def balanced_indices(valid,subject,cap,rng):
    ids=[]
    for s in np.unique(subject[valid]):
        q=np.flatnonzero(valid&(subject==s))
        ids.extend(rng.choice(q,size=min(cap,len(q)),replace=False))
    return np.asarray(ids,np.int32)

def aggregate(vals,subjects,reps=1000):
    frame=pd.DataFrame({'s':subjects,'v':vals}).dropna()
    v=frame.groupby('s').v.mean().to_numpy()
    if len(v)<2:return np.nan,np.nan,np.nan,len(v)
    rng=np.random.default_rng(0)
    draws=v[rng.integers(len(v),size=(reps,len(v)))].mean(1)
    return float(v.mean()),float(np.quantile(draws,.025)),float(np.quantile(draws,.975)),len(v)

def pick_partner(pool,subject,anchor_subject,rng):
    if not len(pool):return None
    for _ in range(16):
        q=int(pool[rng.integers(len(pool))])
        if subject[q]!=anchor_subject:return q
    valid=pool[subject[pool]!=anchor_subject]
    return int(valid[rng.integers(len(valid))]) if len(valid) else None

def direction_metrics(vec,y,valid,subject,cutoffs,subset,subset_name,rep,h,target_name,max_pairs,patient_rows):
    rng=np.random.default_rng(410+h+(0 if target_name=='BIS' else 1))
    norm=np.linalg.norm(vec,axis=1)
    valid=valid&subset&np.isfinite(norm)&(norm>1e-7)
    ix=balanced_indices(valid,subject,CFG['geometry_windows_per_patient'],rng)
    if len(ix)<20:return []
    g=np.searchsorted(cutoffs,y,side='left')
    pools={v:ix[g[ix]==v] for v in range(5)}
    relation={}
    for v in range(5):
        relation[(v,'same')]=pools[v]
        relation[(v,'adjacent')]=np.concatenate([pools[q] for q in range(5) if abs(q-v)==1])
        opposite=[pools[q] for q in range(5) if abs(q-v)>=3]
        relation[(v,'opposite')]=np.concatenate(opposite) if opposite else np.empty(0,np.int32)
        relation[(v,'random')]=ix
    unit=vec[ix]/norm[ix,None]
    lookup={int(q):j for j,q in enumerate(ix)}
    records=[]
    for s in np.unique(subject[ix]):
        own=ix[subject[ix]==s]
        anchors=rng.choice(own,size=min(max_pairs,len(own)),replace=False)
        for i in anchors:
            for cat in ['same','adjacent','opposite','random']:
                q=pick_partner(relation[(g[i],cat)],subject,s,rng)
                if q is not None:
                    sim=float(np.dot(unit[lookup[int(i)]],unit[lookup[q]]))
                    records.append((s,cat,sim))
    frame=pd.DataFrame(records,columns=['subject','category','similarity'])
    out=[]
    for cat in ['same','adjacent','opposite','random']:
        sub=frame[frame.category==cat]
        v,lo,hi,n=aggregate(sub.similarity.to_numpy(),sub.subject.to_numpy(),CFG['bootstrap_replicates'])
        out.append({'subset':subset_name,'representation':rep,'target':target_name,
                    'horizon_seconds':h*10,'category':cat,'cosine':v,'ci_low':lo,'ci_high':hi,
                    'patients':n,'pairs':len(sub)})
    wide=frame.pivot_table(index='subject',columns='category',values='similarity',aggfunc='mean')
    if {'same','opposite'}.issubset(wide):
        pair=wide[['same','opposite']].dropna();v,lo,hi,n=aggregate((pair['same']-pair['opposite']).to_numpy(),pair.index.to_numpy(),CFG['bootstrap_replicates'])
        patient_rows.extend({'subset':subset_name,'representation':rep,'horizon_seconds':h*10,
                             'metric':f'{target_name}_same_minus_opposite_cosine','subject':int(s),'value':float(r.same-r.opposite)}
                            for s,r in pair.iterrows())
        out.append({'subset':subset_name,'representation':rep,'target':target_name,
                    'horizon_seconds':h*10,'category':'same_minus_opposite','cosine':v,
                    'ci_low':lo,'ci_high':hi,'patients':n,'pairs':len(pair)})
    return out

def ordering_metrics(vec,y,valid,subject,subset,subset_name,rep,h,scale,zscale,max_triplets,patient_rows):
    rng=np.random.default_rng(920+h)
    good=valid.all(1)&subset&np.isfinite(vec).all(1)
    ix=balanced_indices(good,subject,CFG['geometry_windows_per_patient'],rng)
    if len(ix)<20:return None,None
    phys=np.asarray(y[ix]/scale,np.float32)
    latent=np.asarray(vec[ix]/zscale,np.float32)
    sub=subject[ix];rec=[]
    for s in np.unique(sub):
        anchors=np.flatnonzero(sub==s)
        others=np.flatnonzero(sub!=s)
        if len(others)<3:continue
        for _ in range(max_triplets):
            i=int(rng.choice(anchors));found=False
            for attempt in range(10):
                j,k=rng.choice(others,size=2,replace=False)
                if sub[j]==sub[k]:continue
                d1=np.linalg.norm(phys[i]-phys[j]);d2=np.linalg.norm(phys[i]-phys[k])
                if abs(d1-d2)>.1:found=True;break
            if not found:continue
            l1=np.linalg.norm(latent[i]-latent[j]);l2=np.linalg.norm(latent[i]-latent[k])
            rec.extend([(s,float(d1),float(l1),int((d1<d2)==(l1<l2))),
                        (s,float(d2),float(l2),int((d1<d2)==(l1<l2)))])
    frame=pd.DataFrame(rec,columns=['subject','d_phys','d_latent','correct'])
    if frame.empty:return None,None
    per=frame.groupby('subject')
    accuracy=per.correct.mean()
    patient_rows.extend({'subset':subset_name,'representation':rep,'horizon_seconds':h*10,
                         'metric':'triplet_accuracy','subject':int(s),'value':float(v)}
                        for s,v in accuracy.items())
    corr=pd.Series({s:spearmanr(q.d_phys,q.d_latent).statistic for s,q in per})
    acc,alo,ahi,na=aggregate(accuracy.to_numpy(),accuracy.index.to_numpy(),CFG['bootstrap_replicates'])
    rho,rlo,rhi,nr=aggregate(corr.to_numpy(),corr.index.to_numpy(),CFG['bootstrap_replicates'])
    row={'subset':subset_name,'representation':rep,'horizon_seconds':h*10,
         'triplet_accuracy':acc,'triplet_ci_low':alo,'triplet_ci_high':ahi,
         'distance_spearman':rho,'spearman_ci_low':rlo,'spearman_ci_high':rhi,
         'patients':na,'triplets':int(len(frame)//2)}
    return row,frame

def main():
    subject=np.asarray(load('test','subject'));subsets=groups()
    assert len(subject)==len(next(iter(subsets.values())))
    scale=np.asarray(list(json.loads((OUT/'response_scale_protocol.json').read_text())['train_300s_change_std'].values()),np.float32)
    assert np.all(scale>0)
    thresholds={};directions=[];order=[];subgroup=[];bin_records=[];patient_rows=[]
    probe=pd.read_csv(OUT/'transition_probe_metrics.csv')
    for k,h in enumerate(HORIZONS):
        y,m=delta_phys('test',k);yt,mt=delta_phys('train',k)
        for j,var in enumerate(['BIS','MAP']):
            thresholds[f'{var}_{h*10}']=np.quantile(yt[mt[:,j],j],[.2,.4,.6,.8]).tolist()
        for rep in ['dz_pred','dz_true']:
            v=dz('test',k,rep);vt=dz('train',k,rep)
            zscale=np.maximum(np.std(vt,axis=0),1e-3)
            for subset_name,subset in subsets.items():
                if subset_name!='overall' and h not in (18,30):continue
                for j,var in enumerate(['BIS','MAP']):
                    cut=np.asarray(thresholds[f'{var}_{h*10}'])
                    out=direction_metrics(v,y[:,j],m[:,j],subject,cut,subset,subset_name,rep,h,var,
                                          CFG['geometry_pairs_per_group_per_patient'] if subset_name=='overall' else 12,patient_rows)
                    directions.extend(out)
                row,frame=ordering_metrics(v,y,m,subject,subset,subset_name,rep,h,scale,zscale,
                              CFG['ordering_triplets_per_patient'] if subset_name=='overall' else 20,patient_rows)
                if row:order.append(row)
                if subset_name=='overall' and h==30 and frame is not None:
                    frame['representation']=rep;bin_records.append(frame[['representation','d_phys','d_latent']])
                if h in (18,30):
                    for j,var in enumerate(VARS):
                        p=np.load(ARRAYS/f'probe_{rep}_{var}_{h}.npy')
                        met=weighted_metrics(y[:,j],p,subject,m[:,j]&subset)
                        lo,hi=patient_bootstrap_stat(y[:,j],p,subject,m[:,j]&subset,'r2',CFG['bootstrap_replicates'])
                        subgroup.append({'subset':subset_name,'representation':rep,'target':var,
                                         'horizon_seconds':h*10,**met,'r2_ci_low':lo,'r2_ci_high':hi})
            print('GEOMETRY',rep,h,flush=True)
    write_json('response_group_protocol.json',{'train_quintile_thresholds':thresholds,
               'pairing':'patient-balanced max 40 windows/patient; partners from other patients',
               'opposite':'quintile index difference >=3','adjacent':'index difference 1',
               'triplet_margin_train_scaled':.1})
    pd.DataFrame(directions).to_csv(OUT/'transition_direction_metrics.csv',index=False)
    pd.DataFrame(order).to_csv(OUT/'response_ordering_metrics.csv',index=False)
    pd.DataFrame(subgroup).to_csv(OUT/'subgroup_grounding_metrics.csv',index=False)
    patient=pd.DataFrame(patient_rows)
    patient.to_csv(OUT/'geometry_patient_summary.csv',index=False)
    paired=[]
    for (subset,h,metric),g in patient.groupby(['subset','horizon_seconds','metric']):
        w=g.pivot_table(index='subject',columns='representation',values='value').dropna()
        if not {'dz_pred','dz_true'}.issubset(w) or len(w)<2:continue
        delta=(w.dz_pred-w.dz_true).to_numpy()
        rng=np.random.default_rng(0)
        draws=delta[rng.integers(len(delta),size=(CFG['bootstrap_replicates'],len(delta)))].mean(1)
        paired.append({'subset':subset,'horizon_seconds':h,'metric':metric,
                       'predicted_minus_factual':float(delta.mean()),
                       'ci_low':float(np.quantile(draws,.025)),'ci_high':float(np.quantile(draws,.975)),
                       'patients':len(delta)})
    pd.DataFrame(paired).to_csv(OUT/'geometry_paired_comparisons.csv',index=False)
    frame=pd.concat(bin_records,ignore_index=True)
    binned=[]
    for rep,g in frame.groupby('representation'):
        edges=np.quantile(g.d_phys,np.linspace(0,1,11));edges=np.unique(edges)
        g=g.assign(bin=pd.cut(g.d_phys,edges,include_lowest=True,duplicates='drop'))
        for b,q in g.groupby('bin',observed=True):
            binned.append({'representation':rep,'phys_distance_midpoint':float(q.d_phys.median()),
                           'latent_distance_median':float(q.d_latent.median()),'pairs':len(q)})
    pd.DataFrame(binned).to_csv(OUT/'response_distance_bins.csv',index=False)
    print('GEOMETRY_COMPLETE',flush=True)

if __name__=='__main__':main()
