"""TRAIN-only state/action regimes and outcome-blind support-cell selection."""
import json, warnings, itertools
import numpy as np
import pandas as pd
from scipy.ndimage import minimum_filter1d,maximum_filter1d
from scipy.optimize import linprog
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import joblib
from common import *

def state_features(store,split):
    idx=np.asarray(arr(split,'index')); result=np.empty((len(idx),40),np.float32)
    for cid in np.unique(idx[:,0]):
        c=store.cases[int(cid)]; where=np.flatnonzero(idx[:,0]==cid); t=idx[where,1]
        x=c['state'].astype(np.float64); valid=np.isfinite(x); parts=[np.nan_to_num(x[t],nan=0),valid[t].astype(float)]
        for n in (6,30):
            # Causal trailing sums via cumulative sums; no future samples enter.
            fill=np.where(valid,x,0); cs=np.vstack([np.zeros((1,3)),np.cumsum(fill,axis=0)])
            c2=np.vstack([np.zeros((1,3)),np.cumsum(fill*fill,axis=0)])
            cv=np.vstack([np.zeros((1,3)),np.cumsum(valid,axis=0)])
            l=np.maximum(t+1-n,0); r=t+1
            count=cv[r]-cv[l]; mean=(cs[r]-cs[l])/np.maximum(count,1)
            var=np.maximum((c2[r]-c2[l])/np.maximum(count,1)-mean**2,0)
            mins=minimum_filter1d(np.where(valid,x,np.inf),size=n,axis=0,mode='constant',origin=(n-1)//2)[t]
            maxs=maximum_filter1d(np.where(valid,x,-np.inf),size=n,axis=0,mode='constant',origin=(n-1)//2)[t]
            # For even n scipy origin=(n-1)//2 produces trailing [t-n+1,t].
            prev=x[np.maximum(t-n,0)]; slope=x[t]-prev
            parts += [np.where(count>0,mean,np.nan),np.sqrt(var),
                      np.where(np.isfinite(mins),mins,np.nan),
                      np.where(np.isfinite(maxs),maxs,np.nan),slope]
        parts.append(np.broadcast_to(c['demographics'],(len(t),4)))
        z=np.concatenate(parts,axis=1)
        assert z.shape[1]==40
        result[where]=z
    return result

def action_summary(a):
    """Fourteen Round-3 schedule statistics, in normalized log-dose units."""
    d=np.diff(a,axis=1)
    return np.concatenate([a.mean(1),a.max(1),a[:,-1],a.sum(1)/30,
        np.abs(d).mean(1),d.mean(1),np.abs(d).max(1)],1).astype('float32')

def cluster_fit(x,ks,label):
    rng=np.random.default_rng(0);sample=rng.choice(len(x),min(10000,len(x)),replace=False)
    choices=[]
    for k in ks:
        km=MiniBatchKMeans(n_clusters=k,random_state=0,batch_size=8192,n_init=5,max_iter=150)
        km.fit(x);lab=km.predict(x);sizes=np.bincount(lab,minlength=k)
        metric=float(silhouette_score(x[sample],lab[sample],sample_size=min(3000,len(sample)),random_state=0))
        choices.append(dict(kind=label,k=k,silhouette=metric,min_cluster=int(sizes.min()),
                            min_fraction=float(sizes.min()/len(x)),model=km,labels=lab))
        print('cluster',label,k,metric,sizes.tolist(),flush=True)
    viable=[r for r in choices if r['min_fraction']>=.005]
    chosen=max(viable or choices,key=lambda r:(r['silhouette'],r['min_fraction']))
    pd.DataFrame([{q:r for q,r in c.items() if q not in ('model','labels')} for c in choices]).to_csv(OUT/f'{label}_cluster_diagnostics.csv',index=False)
    joblib.dump(chosen['model'],OUT/f'{label}_cluster.joblib')
    return chosen

def cell_counts(s,a,blocks,subjects,ks,ka):
    frame=pd.DataFrame({'s':s,'a':a,'block':blocks,'subject':subjects})
    g=frame.groupby(['s','a'])
    b=g['block'].nunique();p=g['subject'].nunique();w=g.size()
    rows=[]
    for i in range(ks):
        for j in range(ka):
            rows.append(dict(state_cluster=i,action_cluster=j,blocks=int(b.get((i,j),0)),
                             patients=int(p.get((i,j),0)),windows=int(w.get((i,j),0))))
    return pd.DataFrame(rows)

def main():
    seed_all(0);store=Store();meta={}
    for split in ('train','val','test'):
        x=state_features(store,split);np.save(ARRAYS/f'{split}_state_raw.npy',x)
    tr=np.load(ARRAYS/'train_state_raw.npy');median=np.nanmedian(np.where(np.isfinite(tr),tr,np.nan),axis=0)
    for split in ('train','val','test'):
        x=np.load(ARRAYS/f'{split}_state_raw.npy');x=np.where(np.isfinite(x),x,median)
        if split=='train':ss=StandardScaler().fit(x)
        np.save(ARRAYS/f'{split}_state_scaled.npy',np.clip(ss.transform(x),-8,8).astype('float32'))
    joblib.dump(dict(median=median,scaler=ss),OUT/'state_scaling.joblib')
    atr=action_summary(np.asarray(arr('train','action')))
    scaler=StandardScaler().fit(atr);joblib.dump(scaler,OUT/'action_scaling.joblib')
    for split in ('train','val','test'):
        a=action_summary(np.asarray(arr(split,'action')))
        np.save(ARRAYS/f'{split}_action_summary.npy',a)
        np.save(ARRAYS/f'{split}_action_scaled.npy',np.clip(scaler.transform(a),-8,8).astype('float32'))
    cs=cluster_fit(np.load(ARRAYS/'train_state_scaled.npy'),CFG['state_k_candidates'],'state')
    ca=cluster_fit(np.load(ARRAYS/'train_action_scaled.npy'),CFG['action_k_candidates'],'action')
    ks=cs['k'];ka=ca['k'];jwrite('cluster_protocol.json',{'state_k':ks,'action_k':ka,
      'selection':'maximum TRAIN-only sampled silhouette among candidates with >=0.5% minimum window cluster',
      'state_features':'current BIS/HR/MAP and availability; trailing 1m and 5m mean/std/min/max/change; age/sex/weight/height',
      'state_train_medians':median.tolist(),'action_features':'Round-3 14-dimensional normalized log-dose summary',
      'train_only_scaling_and_clustering':True,'no_future_physiology_or_CE':True})
    frames={};counts={}
    for split in ('train','val','test'):
        s=cs['model'].predict(np.load(ARRAYS/f'{split}_state_scaled.npy'))
        a=ca['model'].predict(np.load(ARRAYS/f'{split}_action_scaled.npy'))
        np.save(ARRAYS/f'{split}_state_cluster.npy',s.astype('int16'))
        np.save(ARRAYS/f'{split}_action_cluster.npy',a.astype('int16'))
        idx=np.asarray(arr(split,'index'));sub=np.asarray(arr(split,'subject'))
        block=idx[:,0].astype('int64')*100000+idx[:,1]//30
        np.save(ARRAYS/f'{split}_block.npy',block)
        frames[split]=pd.DataFrame({'state':s,'action':a,'caseid':idx[:,0],
                                    'anchor_t':idx[:,1],'subjectid':sub,'block':block})
        counts[split]=cell_counts(s,a,block,sub,ks,ka).rename(columns={c:f'{split}_{c}' for c in ['blocks','patients','windows']})
    matrix=counts['train'].merge(counts['val']).merge(counts['test'])
    # A block may contain several cells near a transition; counts are unique within each cell.
    for i in range(ks):
        for j in range(ka):
            r=matrix[(matrix.state_cluster==i)&(matrix.action_cluster==j)].index[0]
            own=matrix.loc[r,'train_blocks'];sb=matrix[matrix.state_cluster==i].train_blocks.sum()-own
            ab=matrix[matrix.action_cluster==j].train_blocks.sum()-own
            # Distinct marginal patients, not sum of cell patients.
            sf=frames['train'];sp=sf[(sf.state==i)&(sf.action!=j)].subjectid.nunique()
            ap=sf[(sf.action==j)&(sf.state!=i)].subjectid.nunique()
            target=sf[(sf.state==i)&(sf.action==j)]
            contributions=target.groupby('subjectid').block.nunique()
            matrix.loc[r,'other_action_blocks']=sb;matrix.loc[r,'other_state_blocks']=ab
            matrix.loc[r,'other_action_patients']=sp;matrix.loc[r,'other_state_patients']=ap
            matrix.loc[r,'max_patient_block_fraction']=contributions.max()/max(own,1) if len(contributions) else 0
            matrix.loc[r,'train_fraction']=own/frames['train'].block.nunique()
    matrix.to_csv(OUT/'state_action_support_matrix_full.csv',index=False)
    # Deterministic ordered threshold relaxation, then diversified greedy selection.
    threshold=[100,50,200,30]; stages=[]
    for pos,new in [(0,75),(1,30),(2,150),(3,20)]:
        stages.append(tuple(threshold));threshold[pos]=new
    stages.append(tuple(threshold))
    for stage,th in enumerate(stages):
        tb,vb,rb,rp=th
        keep=(matrix.test_blocks>=tb)&(matrix.val_blocks>=vb)&(matrix.train_blocks>=rb)&\
            (matrix.train_patients>=rp)&(matrix.val_patients>=10)&(matrix.test_patients>=15)&\
            (matrix.other_action_blocks>=5*matrix.train_blocks)&(matrix.other_state_blocks>=5*matrix.train_blocks)&\
            (matrix.other_action_patients>=50)&(matrix.other_state_patients>=50)&\
            (matrix.max_patient_block_fraction<=.10)
        if keep.sum()>=4:break
    candidates=matrix[keep].copy()
    if len(candidates)<4:raise RuntimeError(f'Only {len(candidates)} eligible support cells after prescribed relaxation')
    # Prefer well populated 0.5–5% blocks, four distinct state and action
    # clusters, and avoid a selection dominated by high-rate-tail actions.
    train_dose=np.expm1(np.asarray(arr('train','action'))*store.norm['action_std'][None,None,:]+
                          store.norm['action_mean'][None,None,:])/10
    cutoff=np.array(json.loads((SOURCE/'outputs/pump_rate_tail_protocol.json').read_text())['0.99']['maximum_ml_per_10s'])
    tail=(train_dose>cutoff).any((1,2))
    action_tail={j:float(np.mean(tail[frames['train'].action.to_numpy()==j])) for j in range(ka)}
    candidates['preferred_fraction']=candidates.train_fraction.between(.005,.05)
    candidates['action_tail_fraction']=candidates.action_cluster.map(action_tail)
    candidates=candidates.sort_values(['preferred_fraction','test_patients','train_patients','train_blocks','state_cluster','action_cluster'],
                                      ascending=[False,False,False,False,True,True])
    options=[]
    for comb in itertools.combinations(range(len(candidates)),4):
        q=candidates.iloc[list(comb)]
        ns=q.state_cluster.nunique();na=q.action_cluster.nunique()
        score=(-min(ns,na),float(q.action_tail_fraction.sum()),
               -int(q.test_patients.sum()),sum(comb),comb)
        options.append((score,comb))
    if not options:raise RuntimeError('Fewer than four eligible cells')
    _,indices=min(options,key=lambda x:x[0])
    chosen=[(int(r.state_cluster),int(r.action_cluster)) for _,r in candidates.iloc[list(indices)].iterrows()]
    matrix['eligible']=keep;matrix['action_tail_fraction_train_only']=matrix.action_cluster.map(action_tail)
    matrix['selection_rank']=matrix.apply(lambda r:chosen.index((int(r.state_cluster),int(r.action_cluster)))+1 if (int(r.state_cluster),int(r.action_cluster)) in chosen else 0,axis=1)
    matrix['relaxation_stage']=stage;matrix.to_csv(OUT/'support_cell_selection.csv',index=False)
    jwrite('selection_protocol.json',{'thresholds':dict(test_blocks=tb,val_blocks=vb,train_blocks=rb,train_patients=rp,
       val_patients=10,test_patients=15,marginal_block_multiplier=5,marginal_patients=50,max_patient_fraction=.1),
       'relaxation_stage':stage,'selection_order':'exhaustive four-cell combination: maximize distinct state and action clusters, then minimize TRAIN-only high-rate-tail prevalence across action clusters, then maximize TEST patient count; deterministic candidate order tie break',
       'action_tail_fraction_train_only':action_tail,
       'selected_cells':chosen,'test_outcomes_used':False})
    target={split:np.isin(frames[split].state*ka+frames[split].action,[s*ka+a for s,a in chosen]) for split in frames}
    for split in frames:np.save(ARRAYS/f'{split}_target_cell.npy',target[split])
    # Purge every training anchor within +/-30 steps of a removed target anchor.
    train=frames['train'];n=len(train);purge=np.zeros(n,bool);low_purge=np.zeros(n,bool)
    # Retain whole patients for L. Random block retention would leave almost no
    # surviving blocks after the mandated +/-300 s purge of adjacent blocks.
    rng=np.random.default_rng(CFG['selection_seed'])
    tf=train.loc[target['train'],['subjectid','state','action','block']].drop_duplicates()
    counts_by_patient=tf.groupby(['subjectid','state','action']).block.nunique().unstack(['state','action'],fill_value=0)
    counts_by_patient=counts_by_patient.reindex(columns=chosen,fill_value=0)
    desired=.2*counts_by_patient.sum(0).to_numpy()
    patients=counts_by_patient.index.to_numpy();values=counts_by_patient.to_numpy()
    best=None
    for draw in range(2000):
        selected=rng.choice(len(patients),size=max(1,round(.2*len(patients))),replace=False)
        observed=values[selected].sum(0)
        score=float(np.max(np.abs(observed-desired)/np.maximum(desired,1)))
        if best is None or score<best[0]:best=(score,selected,observed)
    retained_patients=set(patients[best[1]].tolist())
    low_removed_target=target['train']&~train.subjectid.isin(retained_patients).to_numpy()
    idx=np.asarray(arr('train','index'))
    for cid in np.unique(idx[:,0]):
        rows=np.flatnonzero(idx[:,0]==cid);t=idx[rows,1]
        targets=t[target['train'][rows]]
        if len(targets):
            distance=np.min(np.abs(t[:,None]-targets[None,:]),axis=1) if len(targets)<150 else None
            if distance is None:
                from scipy.spatial import cKDTree
                distance=cKDTree(targets[:,None]).query(t[:,None])[0]
            purge[rows]=distance<=30
        low_targets=t[low_removed_target[rows]]
        if len(low_targets):
            from scipy.spatial import cKDTree
            low_purge[rows]=cKDTree(low_targets[:,None]).query(t[:,None])[0]<=30
    assert not np.any(target['train']&~purge)
    # Keep retained target blocks in L even if their anchors neighbor removed blocks:
    # the protection requires a clean distance from removed anchor, so resulting
    # support may be <20%; this is measured and disclosed.
    zkeep=~purge;lkeep=~low_purge
    # Random removal: exact window and state/action marginal counts via a
    # capacitated transportation problem over NON-target cells only.
    target_patient_set=set(train.loc[purge,'subjectid'].unique().tolist())
    eligible=np.flatnonzero((~target['train'])&train.subjectid.isin(target_patient_set).to_numpy());need=int(purge.sum())
    non=frames['train'].iloc[eligible].copy();non['index']=eligible
    rng=np.random.default_rng(113);target_removed=frames['train'].iloc[np.flatnonzero(purge)]
    cells=[(s,a) for s in range(ks) for a in range(ka) if (s,a) not in chosen]
    capacity=non.groupby(['state','action']).size()
    A=np.zeros((ks+ka-1,len(cells)),np.float64)
    for col,(s,a) in enumerate(cells):
        A[s,col]=1
        if a<ka-1:A[ks+a,col]=1
    goal=np.r_[np.bincount(target_removed.state,minlength=ks),
               np.bincount(target_removed.action,minlength=ka)[:-1]].astype(float)
    caps=np.array([capacity.get((s,a),0) for s,a in cells])
    lp=linprog(rng.uniform(0,1e-4,len(cells)),A_eq=A,b_eq=goal,
               bounds=list(zip(np.zeros(len(cells)),caps)),method='highs')
    if not lp.success:raise RuntimeError('Marginal-matched non-target removal infeasible: '+lp.message)
    quotas=np.rint(lp.x).astype(int)
    assert np.allclose(A@quotas,goal,atol=1e-3) and quotas.sum()==need
    picked=[]
    unique_blocks=np.sort(non.block.unique());block_rank=dict(zip(unique_blocks,rng.permutation(len(unique_blocks))))
    for (s,a),quota in zip(cells,quotas):
        if quota==0:continue
        group=non[(non.state==s)&(non.action==a)]
        # Draw block groups first; the final block may be partial.
        by_block=group.groupby('block')['index'].apply(np.asarray)
        order=sorted(by_block.index,key=lambda b:block_rank[b]);remaining=int(quota)
        for block in order:
            ids=by_block.loc[block]
            if remaining<=0:break
            if len(ids)<=remaining:picked.extend(ids.tolist());remaining-=len(ids)
            else:picked.extend(rng.choice(ids,size=remaining,replace=False).tolist());remaining=0
        assert remaining==0
    rkeep=np.ones(n,bool);rkeep[picked]=False
    assert rkeep.sum()==zkeep.sum() and np.all(rkeep[target['train']])
    for condition,mask in [('F',np.ones(n,bool)),('L',lkeep),('Z',zkeep),('R',rkeep)]:
        np.save(ARRAYS/f'train_keep_{condition}.npy',mask)
    manifest=pd.DataFrame({'caseid':train.caseid,'anchor_t':train.anchor_t,'subjectid':train.subjectid,
       'block_id':train.block,'state_cluster':train.state,'action_cluster':train.action,
       'target_cell':target['train'],'keep_F':1,'keep_L':lkeep,'keep_Z':zkeep,'keep_R':rkeep})
    manifest.to_csv(OUT/'support_removal_manifest.csv',index=False)
    pools=[]
    for condition,mask in [('F',np.ones(n,bool)),('L',lkeep),('Z',zkeep),('R',rkeep)]:
        part=train[mask];tm=target['train']&mask
        pools.append(dict(condition=condition,windows=int(mask.sum()),removed_windows=int((~mask).sum()),
          blocks=part.block.nunique(),patients=part.subjectid.nunique(),cases=part.caseid.nunique(),
          target_windows=int(tm.sum()),target_blocks=train.loc[tm,'block'].nunique(),
          target_patients=train.loc[tm,'subjectid'].nunique()))
    pd.DataFrame(pools).to_csv(OUT/'training_pool_summary.csv',index=False)
    va=frames['val'];val_trainlike=~target['val'];np.save(ARRAYS/'val_trainlike.npy',val_trainlike)
    np.save(ARRAYS/'val_support_shift.npy',target['val'])
    va[target['val']].to_csv(OUT/'val_support_shift_windows.csv',index=False)
    te=frames['test'].copy();te['state_cluster']=te.state;te['action_cluster']=te.action
    te['target_cell']=target['test'];te[te.target_cell].to_csv(OUT/'target_test_windows.csv',index=False)
    # Cluster descriptions and drug-regime summaries are descriptive and TRAIN-only.
    raw_tr=np.load(ARRAYS/'train_state_raw.npy')
    state_rows=[]
    for s in range(ks):
        m=np.asarray(train.state)==s
        state_rows.append(dict(state_cluster=s,windows=int(m.sum()),blocks=train.loc[m,'block'].nunique(),
          patients=train.loc[m,'subjectid'].nunique(),median_BIS=float(np.nanmedian(raw_tr[m,0])),
          median_HR=float(np.nanmedian(raw_tr[m,1])),median_MAP=float(np.nanmedian(raw_tr[m,2])),
          median_age=float(np.nanmedian(raw_tr[m,-4]))))
    pd.DataFrame(state_rows).to_csv(OUT/'state_cluster_summary.csv',index=False)
    action_rows=[]
    tr_action=np.asarray(arr('train','action'));norm=store.norm
    physical=np.expm1(tr_action*norm['action_std'][None,None,:]+norm['action_mean'][None,None,:])/10
    cutoff=np.array(json.loads((SOURCE/'outputs/pump_rate_tail_protocol.json').read_text())['0.99']['maximum_ml_per_10s'])
    for a in range(ka):
        m=np.asarray(train.action)==a;dose=physical[m]
        action_rows.append(dict(action_cluster=a,windows=int(m.sum()),blocks=train.loc[m,'block'].nunique(),
          patients=train.loc[m,'subjectid'].nunique(),median_ppf_trajectory_json=json.dumps(np.median(dose[:,:,0],axis=0).tolist()),
          median_rft_trajectory_json=json.dumps(np.median(dose[:,:,1],axis=0).tolist()),
          median_ppf_total=float(np.median(dose[:,:,0].sum(1))),median_rft_total=float(np.median(dose[:,:,1].sum(1))),
          q90_ppf_total=float(np.quantile(dose[:,:,0].sum(1),.9)),q90_rft_total=float(np.quantile(dose[:,:,1].sum(1),.9)),
          high_rate_tail_fraction=float(np.mean((dose>cutoff).any((1,2))))))
    pd.DataFrame(action_rows).to_csv(OUT/'action_cluster_summary.csv',index=False)
    jwrite('discovery_summary.json',{'selected_cells':chosen,'cluster_state_k':ks,'cluster_action_k':ka,
       'pool_sizes':pools,'validation_trainlike_windows':int(val_trainlike.sum()),'validation_support_shift_windows':int(target['val'].sum()),
       'test_target_windows':int(target['test'].sum()),'test_target_patients':int(frames['test'].loc[target['test'],'subjectid'].nunique())})
    print('DISCOVERY_COMPLETE',chosen,pools,flush=True)

if __name__=='__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        main()
