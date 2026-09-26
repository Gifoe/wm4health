import json,time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,RandomSampler
from core import *

def loss_fn(pred,b):
    sq=(pred-b['target']).square()*b['target_mask']
    denom=b['target_mask'].sum((0,1)).clamp_min(1)
    # BIS primary, MAP secondary. Standardized target scales are train-only.
    vals=sq.sum((0,1))/denom
    return vals[0]+.25*vals[1]

@torch.no_grad()
def validate(model,loader,store,device):
    model.eval(); sums=np.zeros(2); counts=np.zeros(2)
    scale=torch.tensor(store.norm['state_std'][[0,2]],device=device)
    for b in loader:
        b=to_device(b,device); pred,_=model(b)
        err=(pred-b['target']).abs()*scale*b['target_mask']
        sums+=err.sum((0,1)).cpu().numpy(); counts+=b['target_mask'].sum((0,1)).cpu().numpy()
    return (sums/np.maximum(counts,1)).tolist()

def main():
    seed_all(); store=Store(); train=Windows(store,'train'); val=Windows(store,'val',CFG['validation_stride'])
    device='cuda' if torch.cuda.is_available() else 'cpu'
    out=ROOT/'outputs'; (out/'checkpoints').mkdir(exist_ok=True)
    torch.save({'train_cases':store.splits['train'],'feature_fields':['state','action','demo'],
                'ce_used':False},out/'training_manifest.pt')
    val_loader=DataLoader(val,batch_size=512,num_workers=0,pin_memory=True)
    history=[]; summaries=[]
    for name in MODEL_NAMES:
        checkpoint=out/'checkpoints'/f'{filename(name)}.pt'
        if checkpoint.exists() and (out/f'{filename(name)}_training.json').exists():
            summaries.append(json.loads((out/f'{filename(name)}_training.json').read_text())); continue
        seed_all(); model=model_for(name).to(device)
        count=sum(p.numel() for p in model.parameters())
        optimizer=torch.optim.AdamW(model.parameters(),lr=CFG['learning_rate'],weight_decay=CFG['weight_decay'])
        sampler=RandomSampler(train,replacement=False,num_samples=min(len(train),CFG['training_windows_per_epoch']))
        loader=DataLoader(train,batch_size=CFG['batch_size'],sampler=sampler,num_workers=0,pin_memory=True)
        best=float('inf'); stale=0; start=time.time(); local=[]
        for epoch in range(CFG['epochs']):
            model.train(); running=0.; n=0
            for b in loader:
                b=to_device(b,device); optimizer.zero_grad(set_to_none=True)
                pred,_=model(b); loss=loss_fn(pred,b)
                if not torch.isfinite(loss): raise RuntimeError(f'Nonfinite loss: {name}')
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
                running+=float(loss.detach())*len(pred); n+=len(pred)
            metrics=validate(model,val_loader,store,device)
            row={'model':name,'epoch':epoch+1,'train_loss':running/n,'val_bis_mae':metrics[0],
                 'val_map_mae':metrics[1],'elapsed_seconds':time.time()-start}
            history.append(row); local.append(row); print(json.dumps(row),flush=True)
            pd.DataFrame(history).to_csv(out/'training_history.csv',index=False)
            if metrics[0]<best-1e-4:
                best=metrics[0]; stale=0
                torch.save({'model':model.state_dict(),'name':name,'epoch':epoch+1,
                            'parameters':count,'config':CFG,'validation_bis_mae':best},checkpoint)
            else: stale+=1
            if stale>=CFG['patience']: break
        best_ckpt=torch.load(checkpoint,map_location='cpu',weights_only=False)
        summary={'model':name,'parameters':count,'best_epoch':best_ckpt['epoch'],'epochs_run':len(local),
                 'best_validation_bis_mae':best,'train_windows_available':len(train),
                 'train_windows_per_epoch':min(len(train),CFG['training_windows_per_epoch']),
                 'seconds':time.time()-start,'device':device}
        (out/f'{filename(name)}_training.json').write_text(json.dumps(summary,indent=2)); summaries.append(summary)
        print('TRAINED '+json.dumps(summary),flush=True)
    pd.DataFrame(summaries).to_csv(out/'model_summary.csv',index=False)
    print('TRAINING_COMPLETE',flush=True)

if __name__=='__main__': main()
