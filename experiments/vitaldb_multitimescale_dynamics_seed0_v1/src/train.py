"""Matched seed-0 direct-horizon training with Round-2 permutations."""
import json,time
import numpy as np,pandas as pd,torch
from torch.utils.data import DataLoader,Subset
from models import *

OUT=ROOT/'outputs';CK=ROOT/'checkpoints'
def loss(pred,b):
    target=b['target'][:,[h-1 for h in STEPS]]
    mask=b['target_mask'][:,[h-1 for h in STEPS]]
    sq=(pred-target).square()*mask
    v=sq.sum((0,1))/mask.sum((0,1)).clamp_min(1)
    return v[0]+.25*v[1]

@torch.no_grad()
def validate(model,loader,store,device):
    model.eval();total=np.zeros(2);count=np.zeros(2)
    scale=torch.as_tensor(store.norm['state_std'][[0,2]],device=device)
    for b in loader:
        b=to_device(b,device);p=model(b)
        target=b['target'][:,[h-1 for h in STEPS]]
        mask=b['target_mask'][:,[h-1 for h in STEPS]]
        total+=((p-target).abs()*mask*scale).sum((0,1)).cpu().numpy()
        count+=mask.sum((0,1)).cpu().numpy()
    return total/np.maximum(count,1)

def main():
    seed_all(0);OUT.mkdir(exist_ok=True);CK.mkdir(exist_ok=True)
    store=Store();tr=Windows(store,'train');va=Windows(store,'val',CFG['validation_stride'])
    schedule=np.load(SOURCE/'outputs/epoch_train_indices.npy')
    assert len(tr)==370566 and len(schedule)>=CFG['epochs'] and schedule.shape[1]==80000
    device='cuda' if torch.cuda.is_available() else 'cpu';vl=DataLoader(va,batch_size=512,num_workers=0,pin_memory=True)
    counts={'Direct-MH':parameters(Direct()),'Direct-MH-Capacity':parameters(DirectCapacity()),'MT-Dynamics':parameters(MTDynamics()),'AR-RSSM':parameters(RSSM())}
    assert abs(counts['Direct-MH-Capacity']/counts['MT-Dynamics']-1)<.1
    (OUT/'parameter_counts.json').write_text(json.dumps(counts,indent=2))
    all_rows=[];summary=[]
    for name,cls in [('Direct-MH-Capacity',DirectCapacity),('MT-Dynamics',MTDynamics)]:
        seed_all(0);model=cls().to(device)
        opt=torch.optim.AdamW(model.parameters(),lr=CFG['learning_rate'],weight_decay=CFG['weight_decay'])
        best=np.inf;stale=0;start=time.time();peak=0.;rows=[]
        if device=='cuda':torch.cuda.reset_peak_memory_stats()
        for epoch,ix in enumerate(schedule[:CFG['epochs']],1):
            loader=DataLoader(Subset(tr,ix.tolist()),batch_size=CFG['batch_size'],shuffle=False,num_workers=0,pin_memory=True)
            model.train();s=0.;n=0;gn=0.
            for b in loader:
                b=to_device(b,device);opt.zero_grad(set_to_none=True)
                p=model(b);objective=loss(p,b)
                if not torch.isfinite(objective):raise RuntimeError(f'{name}: nonfinite loss')
                objective.backward();gn+=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1));opt.step()
                s+=float(objective.detach())*len(p);n+=len(p)
            bis,map_=validate(model,vl,store,device)
            if device=='cuda':peak=max(peak,torch.cuda.max_memory_allocated()/2**20)
            rec=dict(model=name,epoch=epoch,train_loss=s/n,val_bis_mae=float(bis),val_map_mae=float(map_),
                     gradient_norm=gn/len(loader),peak_vram_mib=peak,seconds=time.time()-start)
            all_rows.append(rec);rows.append(rec);pd.DataFrame(all_rows).to_csv(OUT/'training_history.csv',index=False)
            print(json.dumps(rec),flush=True)
            if bis<best-1e-4:
                best=float(bis);stale=0
                torch.save(dict(model=model.state_dict(),name=name,epoch=epoch,parameters=counts[name],val_bis_mae=best,val_map_mae=float(map_)),CK/f'{name}.pt')
            else:stale+=1
            if stale>=CFG['patience']:break
        summary.append(dict(model=name,parameters=counts[name],best_epoch=min(rows,key=lambda x:x['val_bis_mae'])['epoch'],
                            epochs_run=len(rows),best_val_bis_mae=best,train_seconds=time.time()-start,
                            peak_vram_mib=peak,train_windows=len(tr),windows_per_epoch=len(schedule[0])))
        pd.DataFrame(summary).to_csv(OUT/'training_summary.csv',index=False)
    print('TRAIN_COMPLETE',flush=True)
if __name__=='__main__':main()
