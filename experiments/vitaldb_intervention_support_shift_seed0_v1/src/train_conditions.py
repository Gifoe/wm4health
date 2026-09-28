"""Train 15 unchanged prospective RSSMs with condition-specific support pools."""
import json,time,hashlib,shutil
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,Subset
from common import *

def train_one(condition,seed,store,train,val,device):
    seed_all(seed);start=time.time()
    model=RSSM().to(device);model.set_action_stats(store)
    optimizer=torch.optim.AdamW(model.parameters(),lr=CFG['learning_rate'],weight_decay=CFG['weight_decay'])
    available=np.flatnonzero(np.load(ARRAYS/f'train_keep_{condition}.npy'))
    pool_sha=hashlib.sha256(np.load(ARRAYS/f'train_keep_{condition}.npy').tobytes()).hexdigest()
    val_sha=hashlib.sha256(np.load(ARRAYS/'val_trainlike.npy').tobytes()).hexdigest()
    assert len(available)>=CFG['training_windows_per_epoch']
    rng=np.random.default_rng(seed)
    schedules=[rng.permutation(available)[:CFG['training_windows_per_epoch']] for _ in range(CFG['epochs'])]
    vl=DataLoader(val,batch_size=512,num_workers=0,pin_memory=True)
    best=np.inf;stale=0;history=[]
    for epoch,indices in enumerate(schedules,1):
        loader=DataLoader(Subset(train,indices.tolist()),batch_size=CFG['batch_size'],shuffle=False,
                          num_workers=0,pin_memory=True)
        model.train();total=0.;n=0
        for b in loader:
            b=to_device(b,device);optimizer.zero_grad(set_to_none=True)
            pred,_,_=model(b,'true');loss=loss_fn(pred,b)
            if not torch.isfinite(loss):raise RuntimeError(f'{condition}/{seed}: nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
            total+=float(loss.detach())*len(pred);n+=len(pred)
        score=validate(model,vl,store,device,'true')
        rec=dict(condition=condition,seed=seed,epoch=epoch,train_loss=total/n,
                 val_trainlike_bis_mae=float(score[0]),val_trainlike_map_mae=float(score[1]),seconds=time.time()-start)
        history.append(rec);pd.DataFrame(history).to_csv(OUT/f'training_history_{condition}_seed{seed}.csv',index=False)
        print('EPOCH',json.dumps(rec),flush=True)
        if score[0]<best-1e-4:
            best=float(score[0]);stale=0
            torch.save({'model':model.state_dict(),'policy':'true','name':'Prospective RSSM',
                        'condition':condition,'seed':seed,'epoch':epoch,
                        'train_pool_sha256':pool_sha,'val_trainlike_sha256':val_sha,
                        'parameters':sum(p.numel() for p in model.parameters()),
                        'val_trainlike_bis_mae':best},OUT/'checkpoints'/f'{condition}_seed{seed}.pt')
        else:stale+=1
        if stale>=CFG['patience']:break
    return dict(condition=condition,seed=seed,reused_round3=False,parameters=sum(p.numel() for p in model.parameters()),
      best_epoch=torch.load(OUT/'checkpoints'/f'{condition}_seed{seed}.pt',map_location='cpu',weights_only=False)['epoch'],
      epochs_run=epoch,best_val_trainlike_bis_mae=best,train_windows_available=len(available),
      windows_per_epoch=CFG['training_windows_per_epoch'],seconds=time.time()-start)

def main():
    seed_all(0);(OUT/'checkpoints').mkdir(exist_ok=True)
    store=Store();train=Windows(store,'train');val_all=Windows(store,'val')
    assert np.array_equal(train.indices,arr('train','index'))
    assert np.array_equal(val_all.indices,arr('val','index'))
    val=Subset(val_all,np.flatnonzero(np.load(ARRAYS/'val_trainlike.npy')[::CFG['validation_stride']])*CFG['validation_stride'])
    assert len(val)>0
    device='cuda' if torch.cuda.is_available() else 'cpu'
    assert device=='cuda','GPU required for 15-model fixed benchmark'
    rows=[]
    for seed in CFG['seeds']:
        source=ROUND3/'outputs/checkpoints'/f'seed{seed}.pt';dest=OUT/'checkpoints'/f'F_seed{seed}.pt'
        shutil.copy2(source,dest)
        ck=torch.load(dest,map_location='cpu',weights_only=False)
        assert ck['parameters']==58946
        assert hashlib.sha256(source.read_bytes()).digest()==hashlib.sha256(dest.read_bytes()).digest()
        rows.append(dict(condition='F',seed=seed,reused_round3=True,parameters=58946,
          best_epoch=ck['epoch'],epochs_run=np.nan,best_val_trainlike_bis_mae=np.nan,
          train_windows_available=len(train),windows_per_epoch=CFG['training_windows_per_epoch'],seconds=0))
    pd.DataFrame(rows).to_csv(OUT/'ensemble_model_summary.csv',index=False)
    for condition in ('L','Z','R'):
        for seed in CFG['seeds']:
            dest=OUT/'checkpoints'/f'{condition}_seed{seed}.pt'
            if dest.exists():
                ck=torch.load(dest,map_location='cpu',weights_only=False)
                expected_pool=hashlib.sha256(np.load(ARRAYS/f'train_keep_{condition}.npy').tobytes()).hexdigest()
                expected_val=hashlib.sha256(np.load(ARRAYS/'val_trainlike.npy').tobytes()).hexdigest()
                assert ck.get('train_pool_sha256')==expected_pool and ck.get('val_trainlike_sha256')==expected_val,\
                    f'Stale checkpoint for {condition} seed{seed}: support manifest changed'
                if any(r['condition']==condition and r['seed']==seed for r in rows):continue
                rows.append(dict(condition=condition,seed=seed,reused_round3=False,
                   parameters=ck['parameters'],best_epoch=ck['epoch'],epochs_run=np.nan,
                   best_val_trainlike_bis_mae=ck['val_trainlike_bis_mae'],
                   train_windows_available=int(np.load(ARRAYS/f'train_keep_{condition}.npy').sum()),
                   windows_per_epoch=CFG['training_windows_per_epoch'],seconds=np.nan))
                continue
            row=train_one(condition,seed,store,train,val,device)
            rows.append(row);pd.DataFrame(rows).to_csv(OUT/'ensemble_model_summary.csv',index=False)
            print('TRAINED',condition,seed,flush=True)
    pd.DataFrame(rows).to_csv(OUT/'ensemble_model_summary.csv',index=False)
    print('TRAIN_CONDITIONS_COMPLETE',flush=True)

if __name__=='__main__':main()
