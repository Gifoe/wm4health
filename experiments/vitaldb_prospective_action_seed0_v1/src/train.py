"""Matched initialization, samples, optimizer and validation; rollout action is the only model difference."""
import json,time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,Subset
from core import *

def loss_fn(pred,b):
    sq=(pred-b['target']).square()*b['target_mask']
    vals=sq.sum((0,1))/b['target_mask'].sum((0,1)).clamp_min(1)
    return vals[0]+.25*vals[1]

@torch.no_grad()
def validate(model,loader,store,device,policy):
    model.eval(); sums=np.zeros(2); counts=np.zeros(2)
    scale=torch.as_tensor(store.norm['state_std'][[0,2]],device=device)
    for b in loader:
        b=to_device(b,device); pred,_,_=model(b,policy)
        error=(pred-b['target']).abs()*scale*b['target_mask']
        sums+=error.sum((0,1)).cpu().numpy(); counts+=b['target_mask'].sum((0,1)).cpu().numpy()
    return sums/np.maximum(counts,1)

def main():
    seed_all(); store=Store(); train=Windows(store,'train'); val=Windows(store,'val',CFG['validation_stride'])
    out=ROOT/'outputs';(out/'checkpoints').mkdir(parents=True,exist_ok=True)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    val_loader=DataLoader(val,batch_size=512,num_workers=0,pin_memory=True)
    # One fixed random permutation per epoch, shared exactly across conditions.
    rng=np.random.default_rng(CFG['seed'])
    schedules=[rng.permutation(len(train))[:CFG['training_windows_per_epoch']].copy() for _ in range(CFG['epochs'])]
    np.save(out/'epoch_train_indices.npy',np.stack(schedules))
    rows=[]; summaries=[]
    for name,policy in [('Historical RSSM','hold'),('Prospective RSSM','true')]:
        seed_all(); model=RSSM().to(device); model.set_action_stats(store)
        opt=torch.optim.AdamW(model.parameters(),lr=CFG['learning_rate'],weight_decay=CFG['weight_decay'])
        best=np.inf; stale=0; start=time.time(); local=[]
        for epoch,idx in enumerate(schedules,1):
            loader=DataLoader(Subset(train,idx.tolist()),batch_size=CFG['batch_size'],shuffle=False,num_workers=0,pin_memory=True)
            model.train(); total=0.; n=0
            for b in loader:
                b=to_device(b,device);opt.zero_grad(set_to_none=True)
                pred,_,_=model(b,policy);loss=loss_fn(pred,b)
                if not torch.isfinite(loss): raise RuntimeError(name+' nonfinite loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
                total+=float(loss.detach())*len(pred);n+=len(pred)
            scores=validate(model,val_loader,store,device,policy)
            row={'model':name,'epoch':epoch,'train_loss':total/n,'val_bis_mae':scores[0],
                 'val_map_mae':scores[1],'seconds':time.time()-start}
            rows.append(row);local.append(row);pd.DataFrame(rows).to_csv(out/'training_history.csv',index=False)
            print(json.dumps(row),flush=True)
            if scores[0]<best-1e-4:
                best=scores[0];stale=0
                torch.save({'model':model.state_dict(),'policy':policy,'name':name,'epoch':epoch,
                            'parameters':sum(p.numel() for p in model.parameters()),'val_bis_mae':best},
                           out/'checkpoints'/f'{policy}.pt')
            else: stale+=1
            if stale>=CFG['patience']: break
        ck=torch.load(out/'checkpoints'/f'{policy}.pt',map_location='cpu',weights_only=False)
        summaries.append({'model':name,'policy':policy,'parameters':ck['parameters'],
                          'best_epoch':ck['epoch'],'epochs_run':len(local),'best_val_bis_mae':best,
                          'train_windows':len(train),'windows_per_epoch':len(schedules[0]),'seconds':time.time()-start})
    pd.DataFrame(summaries).to_csv(out/'model_summary.csv',index=False)
    print('TRAIN_COMPLETE',flush=True)

if __name__=='__main__': main()
