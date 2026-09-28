"""Horizon-matched predicted-transition support from TRAIN rollouts only."""
import json
import faiss
import numpy as np
import pandas as pd
from common import *

faiss.omp_set_num_threads(8)

def query(index,x):
    out=[]
    for start in range(0,len(x),4096):
        d,_=index.search(np.ascontiguousarray(x[start:start+4096]),20)
        out.append(np.sqrt(d[:,19]))
    return np.concatenate(out)

def main():
    sample=np.load(ARRAYS/'train_dynamic_sample_index.npy')
    z=np.load(ARRAYS/'train_dynamic_steps.npy',mmap_mode='r')
    a=np.load(ARRAYS/'train_action.npy',mmap_mode='r')[sample]
    results={s:np.empty((len(np.load(ARRAYS/f'{s}_index.npy')),F),np.float32) for s in ['val','test']}
    train_scale=[]
    for h in range(F):
        train_z=np.asarray(z[:,h],np.float32);train_a=np.asarray(a[:,h],np.float32)
        zm=train_z.mean(0);zs=np.maximum(train_z.std(0),.001)
        am=train_a.mean(0);ast=np.maximum(train_a.std(0),.001)
        def transform(zz,aa):
            zz=np.clip((zz-zm)/zs,-8,8)/np.sqrt(64)
            aa=np.clip((aa-am)/ast,-8,8)/np.sqrt(2)
            return np.ascontiguousarray(np.concatenate([zz,aa],1).astype('float32'))
        train=transform(train_z,train_a)
        idx=faiss.IndexIVFFlat(faiss.IndexFlatL2(66),66,128,faiss.METRIC_L2)
        idx.train(train);idx.add(train);idx.nprobe=24
        self_d=query(idx,train)
        scale=float(np.quantile(self_d,.95));train_scale.append(scale)
        for split in ['val','test']:
            qz=np.load(ARRAYS/f'{split}_steps.npy',mmap_mode='r')[:,h]
            qa=np.load(ARRAYS/f'{split}_action.npy',mmap_mode='r')[:,h]
            results[split][:,h]=query(idx,transform(qz,qa))/max(scale,.001)
        print('dynamic horizon',h+1,'scale',scale,flush=True)
    for split,score in results.items():np.save(ARRAYS/f'{split}_dynamic_support.npy',score)
    jwrite('dynamic_support_protocol.json',{'reference_train_windows':len(sample),
      'horizon_matched':True,'features':'seed0 predicted z_h and factual future action a_h, each block TRAIN-standardized',
      'distance':'20th TRAIN neighbor; scaled by TRAIN 95th percentile of within-bank 20th-neighbor distance at same h',
      'faiss':'IVFFlat nlist128 nprobe24','train_self_distance_q95_by_horizon':train_scale,
      'no_future_physiology_or_ce_in_support':True})
    print('DYNAMIC_COMPLETE',flush=True)

if __name__=='__main__':main()
