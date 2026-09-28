"""Empirical block/cell support and separate marginal familiarity checks."""
import json
import numpy as np
import pandas as pd
import faiss
from common import *

faiss.omp_set_num_threads(8)

def index(bank,seed=0):
    bank=np.ascontiguousarray(bank,dtype='float32');dim=bank.shape[1]
    nlist=256;idx=faiss.IndexIVFFlat(faiss.IndexFlatL2(dim),dim,nlist,faiss.METRIC_L2)
    rng=np.random.default_rng(seed)
    idx.train(bank[rng.choice(len(bank),size=min(50000,len(bank)),replace=False)])
    idx.add(bank);idx.nprobe=24
    return idx

def main():
    train_idx=np.asarray(arr('train','index'));test_idx=np.asarray(arr('test','index'))
    tr_s=np.load(ARRAYS/'train_state_cluster.npy');tr_a=np.load(ARRAYS/'train_action_cluster.npy')
    te_s=np.load(ARRAYS/'test_state_cluster.npy');te_a=np.load(ARRAYS/'test_action_cluster.npy')
    target=np.load(ARRAYS/'test_target_cell.npy');tblock=np.load(ARRAYS/'train_block.npy')
    tr_sub=np.asarray(arr('train','subject'));te_sub=np.asarray(arr('test','subject'))
    state=np.load(ARRAYS/'train_state_scaled.npy',mmap_mode='r');action=np.load(ARRAYS/'train_action_scaled.npy',mmap_mode='r')
    tr_flat=np.asarray(arr('train','action')).reshape(len(tr_s),-1)
    flat_mean=tr_flat.mean(0);flat_std=np.maximum(tr_flat.std(0),1e-3)
    flat=np.clip((tr_flat-flat_mean)/flat_std,-8,8).astype('float32')
    qflat=np.clip((np.asarray(arr('test','action'))[target].reshape(int(target.sum()),-1)-flat_mean)/flat_std,-8,8).astype('float32')
    qstate=np.ascontiguousarray(np.load(ARRAYS/'test_state_scaled.npy',mmap_mode='r')[target])
    qaction=np.ascontiguousarray(np.load(ARRAYS/'test_action_scaled.npy',mmap_mode='r')[target])
    qsub=te_sub[target];qs=te_s[target];qa=te_a[target]
    chosen=[tuple(x) for x in json.loads((OUT/'selection_protocol.json').read_text())['selected_cells']]
    cell=[];fam=[]
    for condition in ('F','L','Z','R'):
        keep=np.load(ARRAYS/f'train_keep_{condition}.npy')
        frame=pd.DataFrame({'state':tr_s[keep],'action':tr_a[keep],'block':tblock[keep],
                            'subject':tr_sub[keep]})
        g=frame.groupby(['state','action']);ks=int(tr_s.max())+1;ka=int(tr_a.max())+1
        blocks=g.block.nunique();patients=g.subject.nunique()
        for sn,an in chosen:
            count=int(blocks.get((sn,an),0));sm=int(frame[frame.state==sn].block.nunique())
            am=int(frame[frame.action==an].block.nunique())
            cell.append(dict(condition=condition,state_cluster=sn,action_cluster=an,
              cell_train_blocks=count,cell_train_patients=int(patients.get((sn,an),0)),
              state_marginal_train_blocks=sm,action_marginal_train_blocks=am,
              smoothed_conditional_action_probability=(count+1)/(sm+ka),
              target_test_windows=int(np.sum((qs==sn)&(qa==an))),
              target_test_patients=len(np.unique(qsub[(qs==sn)&(qa==an)]))))
        bank_s=np.ascontiguousarray(state[keep]);bank_a=np.ascontiguousarray(action[keep])
        si=index(bank_s,0);ai=index(bank_a,1);fi=index(flat[keep],2)
        sd,sneighbors=si.search(qstate,100);ad,_=ai.search(qaction,20)
        fd,_=fi.search(np.ascontiguousarray(qflat),20)
        conditional=np.empty(len(qstate),np.float32)
        for i in range(0,len(qstate),512):
            en=min(i+512,len(qstate))
            delta=bank_a[sneighbors[i:en]]-qaction[i:en,None,:]
            conditional[i:en]=np.linalg.norm(delta,axis=2).min(1)
        for sn,an in chosen:
            m=(qs==sn)&(qa==an)
            fam.append(dict(condition=condition,state_cluster=sn,action_cluster=an,
               state_knn20_distance=float(np.median(np.sqrt(sd[m,19]))),
               action_knn20_distance=float(np.median(np.sqrt(ad[m,19]))),
               action_flat_knn20_distance=float(np.median(np.sqrt(fd[m,19]))),
               conditional_action_distance=float(np.median(conditional[m])),
               n_windows=int(m.sum()),n_patients=len(np.unique(qsub[m]))))
        np.save(ARRAYS/f'target_state_knn_{condition}.npy',np.sqrt(sd[:,19]))
        np.save(ARRAYS/f'target_action_knn_{condition}.npy',np.sqrt(ad[:,19]))
        np.save(ARRAYS/f'target_action_flat_knn_{condition}.npy',np.sqrt(fd[:,19]))
        np.save(ARRAYS/f'target_conditional_action_{condition}.npy',conditional)
        print('SUPPORT_DIAGNOSTIC',condition,flush=True)
    cell=pd.DataFrame(cell);fam=pd.DataFrame(fam)
    cell.to_csv(OUT/'cell_support_by_condition.csv',index=False)
    fam.to_csv(OUT/'marginal_familiarity_checks.csv',index=False)
    # The condition changes the neighbor bank, but no outcome values are read.
    jwrite('support_diagnostic_protocol.json',{'index':'FAISS IVFFlat nlist256 nprobe24',
       'train_only_regime_and_scaling':True,'conditional_action':'minimum scaled 14-feature action distance among 100 nearest scaled raw historical-state summaries',
       'marginal_action_distances':'20th neighbor in 14-feature summary and separate 60-dimensional flattened trajectory, each standardized on TRAIN',
       'cell_support':'unique 5-minute TRAIN blocks; Laplace +1 conditional action smoothing',
       'future_outcomes_or_CE_used':False})
    print('SUPPORT_DIAGNOSTICS_COMPLETE',flush=True)

if __name__=='__main__':main()
