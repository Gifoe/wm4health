"""Train seed-0 direct controls with the original Round-2 schedules and optimizer."""
import json,time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,Subset
from common import *
from direct import Direct

def loss(pred,target,mask):
    sq=(pred-target).square()*mask
    v=sq.sum((0,1))/mask.sum((0,1)).clamp_min(1)
    return v[0]+.25*v[1]

@torch.no_grad()
def validate(model,loader,store,device,steps):
    model.eval();total=np.zeros(2);counts=np.zeros(2)
    scale=torch.as_tensor(store.norm['state_std'][[0,2]],device=device)
    for b in loader:
        b=to_device(b,device);p=model(b,steps)
        t=b['target'] if steps is None else b['target'][:,[h-1 for h in steps]]
        m=b['target_mask'] if steps is None else b['target_mask'][:,[h-1 for h in steps]]
        total+=((p-t).abs()*m*scale).sum((0,1)).cpu().numpy()
        counts+=m.sum((0,1)).cpu().numpy()
    return total/np.maximum(counts,1)

def main():
    seed_all(0);OUT.mkdir(exist_ok=True);(OUT/'checkpoints').mkdir(exist_ok=True)
    store=Store();train=Windows(store,'train');val=Windows(store,'val',CFG['validation_stride'])
    schedules=np.load(SOURCE/'outputs/epoch_train_indices.npy')
    assert len(train)==370566 and len(schedules)>=CFG['epochs'] and schedules.shape[1]==80000
    device='cuda' if torch.cuda.is_available() else 'cpu'
    vl=DataLoader(val,batch_size=512,num_workers=0,pin_memory=True)
    summaries=[]
    for name,steps in [('Direct-MH',HORIZONS),('Direct-Traj',None)]:
        seed_all(0);model=Direct().to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=CFG['learning_rate'],weight_decay=CFG['weight_decay'])
        best=float('inf');stale=0;start=time.time();rows=[]
        for epoch,idx in enumerate(schedules[:CFG['epochs']],1):
            loader=DataLoader(Subset(train,idx.tolist()),batch_size=CFG['batch_size'],shuffle=False,num_workers=0,pin_memory=True)
            model.train();total=0.;n=0
            for b in loader:
                b=to_device(b,device);optimizer.zero_grad(set_to_none=True)
                p=model(b,steps)
                t=b['target'] if steps is None else b['target'][:,[h-1 for h in steps]]
                m=b['target_mask'] if steps is None else b['target_mask'][:,[h-1 for h in steps]]
                objective=loss(p,t,m)
                if not torch.isfinite(objective):raise RuntimeError('Nonfinite direct loss')
                objective.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
                total+=float(objective.detach())*len(p);n+=len(p)
            score=validate(model,vl,store,device,steps)
            rec={'model':name,'epoch':epoch,'train_loss':total/n,'val_bis_mae':score[0],
                 'val_map_mae':score[1],'seconds':time.time()-start}
            rows.append(rec);pd.DataFrame(rows).to_csv(OUT/f'training_history_{name}.csv',index=False)
            print(json.dumps(rec),flush=True)
            if score[0]<best-1e-4:
                best=float(score[0]);stale=0
                torch.save({'model':model.state_dict(),'name':name,'steps':steps,'epoch':epoch,
                            'parameters':sum(p.numel() for p in model.parameters()),'val_bis_mae':best},
                           OUT/'checkpoints'/f'{name}.pt')
            else:stale+=1
            if stale>=CFG['patience']:break
        summaries.append({'model':name,'parameters':sum(p.numel() for p in model.parameters()),
                          'best_epoch':int(pd.DataFrame(rows).sort_values('val_bis_mae').iloc[0].epoch),
                          'epochs_run':len(rows),'best_val_bis_mae':best,'seconds':time.time()-start,
                          'train_windows':len(train),'windows_per_epoch':len(schedules[0])})
    pd.DataFrame(summaries).to_csv(OUT/'training_summary.csv',index=False)
    print('TRAIN_COMPLETE',flush=True)

if __name__=='__main__':main()
