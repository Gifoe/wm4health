"""Freeze the five RSSMs; extract factual predictions and TRAIN-only support bank."""
import hashlib,json,time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,Subset
from common import *

def physical(p,store):
    return p*store.norm['state_std'][[0,2]]+store.norm['state_mean'][[0,2]]

@torch.no_grad()
def extract(ds,store,model,split,seed,predict=True):
    model.eval();device=next(model.parameters()).device
    loader=DataLoader(ds,batch_size=512,num_workers=0,pin_memory=True)
    z=[];action=[];pred=[];steps=[];target=[];mask=[];current=[];raw=[]
    for j,b in enumerate(loader):
        action.append(b['future_action'][:,:,:2].numpy().astype('float32'))
        if seed==0:
            target.append(physical(b['target'].numpy(),store).astype('float32'))
            mask.append(b['target_mask'][:,:,0].numpy().astype('bool'))
            current.append(b['state'][:,-1,:3].numpy().astype('float32'))
            raw.append(b['demo'].numpy().astype('float32'))
        b=to_device(b,device)
        if predict:
            p,zz,ss=model(b,'true',return_steps=(seed==0))
            pred.append(physical(p.cpu().numpy(),store).astype('float32'))
            if seed==0:steps.append(ss.cpu().numpy().astype('float32'))
        else:
            x=torch.cat([b['state'],b['action'],b['demo'][:,None,:].expand(-1,H,-1)],-1)
            _,h=model.encoder(x);zz=model.latent(h[-1])
        if seed==0:z.append(zz.cpu().numpy().astype('float32'))
        if (j+1)%100==0:print('extract',split,seed,j+1,flush=True)
    if seed==0:
        np.save(ARRAYS/f'{split}_z.npy',np.concatenate(z))
        np.save(ARRAYS/f'{split}_action.npy',np.concatenate(action))
        np.save(ARRAYS/f'{split}_target.npy',np.concatenate(target))
        np.save(ARRAYS/f'{split}_mask.npy',np.concatenate(mask))
        np.save(ARRAYS/f'{split}_current.npy',np.concatenate(current))
        np.save(ARRAYS/f'{split}_demo.npy',np.concatenate(raw))
        np.save(ARRAYS/f'{split}_index.npy',ds.indices)
        np.save(ARRAYS/f'{split}_subject.npy',np.array([int(store.cases[int(cid)]['subjectid']) for cid,t in ds.indices],np.int32))
        if predict:np.save(ARRAYS/f'{split}_steps.npy',np.concatenate(steps))
    if predict:np.save(ARRAYS/f'{split}_pred_seed{seed}.npy',np.concatenate(pred))

def main():
    seed_all(0);ARRAYS.mkdir(parents=True,exist_ok=True)
    store=Store();device='cuda' if torch.cuda.is_available() else 'cpu'
    for split in ['train','val','test']:
        ds=Windows(store,split)
        model=model_for(0,store,device)
        extract(ds,store,model,split,0,predict=(split!='train'))
        del model;torch.cuda.empty_cache()
        if split!='train':
            for seed in CFG['seeds'][1:]:
                model=model_for(seed,store,device)
                extract(ds,store,model,split,seed)
                del model;torch.cuda.empty_cache()
        # Verify alignment with original Round-2 per-window reference on TEST.
        if split=='test':
            ref=np.load(SOURCE/'outputs/test_reference.npz')
            idx=np.load(ARRAYS/'test_index.npy')
            assert np.array_equal(idx[:,0],ref['case']) and np.array_equal(idx[:,1],ref['t'])
            np.save(ARRAYS/'test_subject.npy',ref['subject'])
            for k in ['large_label','upcoming_label','divergence','quartile']:
                np.save(ARRAYS/f'test_{k}.npy',ref[k])
    # Horizon-matched TRAIN rollout bank, fixed action-independent random sample.
    # Full static support bank above still contains every eligible TRAIN window.
    train_ds=Windows(store,'train');rng=np.random.default_rng(0)
    picked=np.sort(rng.choice(len(train_ds),size=min(50000,len(train_ds)),replace=False))
    np.save(ARRAYS/'train_dynamic_sample_index.npy',picked)
    model=model_for(0,store,device);dynamic=[]
    for j,b in enumerate(DataLoader(Subset(train_ds,picked.tolist()),batch_size=512,num_workers=0,pin_memory=True)):
        with torch.no_grad():
            _,_,ss=model(to_device(b,device),'true',return_steps=True)
        dynamic.append(ss.cpu().numpy().astype('float32'))
        if j%50==0:print('dynamic train rollout',j,flush=True)
    np.save(ARRAYS/'train_dynamic_steps.npy',np.concatenate(dynamic))
    del model;torch.cuda.empty_cache()
    train_idx=np.load(ARRAYS/'train_index.npy');tz=np.load(ARRAYS/'train_z.npy',mmap_mode='r')
    # The next eligible anchor is the factual next observed history, when present.
    lookup={(int(cid),int(t)):i for i,(cid,t) in enumerate(train_idx)}
    i=np.array([j for j,(cid,t) in enumerate(train_idx) if (int(cid),int(t)+1) in lookup],np.int32)
    nxt=np.array([lookup[(int(train_idx[j,0]),int(train_idx[j,1])+1)] for j in i],np.int32)
    np.save(ARRAYS/'train_transition_index.npy',np.stack([i,nxt],1))
    checks={'source_train_windows':len(train_idx),'train_adjacent_transitions':len(i),
      'train_next_latent_from_next_factual_history':True,'source_seed0_checkpoint_sha256':hashlib.sha256((SOURCE/'outputs/checkpoints/true.pt').read_bytes()).hexdigest(),
      'copied_seed0_checkpoint_sha256':hashlib.sha256((OUT/'checkpoints/seed0.pt').read_bytes()).hexdigest(),
      'train_dynamic_sample_windows':len(picked),'dynamic_reference_horizon_matched':True,
      'ce_present_in_support_features':False,'future_physiology_present_in_support_features':False}
    jwrite('reference_integrity.json',checks)
    print('REFERENCE_COMPLETE',checks,flush=True)

if __name__=='__main__':main()
