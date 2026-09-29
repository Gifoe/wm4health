"""Matched seed-0 training, using Round-2 backbone and epoch permutations."""
import json,sys,time,shutil
from pathlib import Path
import numpy as np,pandas as pd,torch,yaml
from torch import nn
from torch.utils.data import Dataset,DataLoader,Subset

ROOT=Path(__file__).resolve().parents[1]
C=yaml.safe_load((ROOT/'config.yaml').read_text())
SOURCE=(ROOT/C['source_experiment']).resolve()
AUDIT=(ROOT/C['audit_experiment']).resolve()
sys.path.insert(0,str(SOURCE/'src'))
from core import Store,Windows,RSSM,to_device,seed_all
from train import loss_fn as original_loss_fn

HORIZONS=np.asarray(C['horizons'])
SCALES=np.asarray(list(json.loads((AUDIT/'outputs/response_scale_protocol.json').read_text())['train_300s_change_std'].values()),np.float32)

class GroundWindows(Dataset):
    def __init__(self,base):self.base=base
    def __len__(self):return len(self.base)
    def __getitem__(self,i):
        b=self.base[i];cid,t=self.base.indices[i];c=self.base.store.cases[int(cid)]
        cur=c['state'][t];f=c['state'][t+HORIZONS]
        cm=c['state_fresh'][t].astype(bool);fm=c['state_fresh'][t+HORIZONS].astype(bool)
        b['ground_current']=np.nan_to_num(c['sn'][t],nan=0).astype(np.float32)
        b['ground_current_mask']=(cm&np.isfinite(cur)).astype(np.float32)
        delta=(f-cur)/SCALES
        b['ground_delta']=np.nan_to_num(delta,nan=0,posinf=0,neginf=0).astype(np.float32)
        b['ground_delta_mask']=(fm&cm[None,:]&np.isfinite(f)&np.isfinite(cur)[None,:]).astype(np.float32)
        return b

def aux_loss(head,z0,roll,b,kind):
    if kind=='state':pred=head(z0);y=b['ground_current'];mask=b['ground_current_mask']
    else:
        pred=head(roll[:,HORIZONS-1]-z0[:,None,:]);y=b['ground_delta'];mask=b['ground_delta_mask']
    component=((pred-y).square()*mask).sum(tuple(range(pred.ndim-1)))/mask.sum(tuple(range(mask.ndim-1))).clamp_min(1)
    return component.mean(),pred

@torch.no_grad()
def validate(model,loader,store,device):
    model.eval();s=np.zeros(2);n=np.zeros(2)
    scale=torch.as_tensor(store.norm['state_std'][[0,2]],device=device)
    for b in loader:
        b=to_device(b,device);pred,_,_=model(b,'true')
        e=(pred-b['target']).abs()*scale*b['target_mask']
        s+=e.sum((0,1)).cpu().numpy();n+=b['target_mask'].sum((0,1)).cpu().numpy()
    return s/np.maximum(n,1)

def main():
    seed_all(0);store=Store();train=GroundWindows(Windows(store,'train'))
    val=Windows(store,'val',C['validation_stride'])
    out=ROOT/'outputs';out.mkdir(exist_ok=True);(ROOT/'checkpoints').mkdir(exist_ok=True)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    vloader=DataLoader(val,batch_size=512,num_workers=0,pin_memory=True)
    schedule=np.load(SOURCE/'outputs/epoch_train_indices.npy')
    assert schedule.shape[0]>=C['epochs'] and schedule.max()<len(train)
    counts=[];rows=[];summary=[]
    base=RSSM();nbase=sum(p.numel() for p in base.parameters())
    for kind in ('state','transition'):
        for lam in C['lambda_grid']:
            seed_all(0)
            model=RSSM().to(device);model.set_action_stats(store)
            head=nn.Sequential(nn.Linear(64,32),nn.SiLU(),nn.Linear(32,3)).to(device)
            nh=sum(p.numel() for p in head.parameters())
            counts.append(dict(model=kind,lambda_weight=lam,backbone_parameters=nbase,training_head_parameters=nh,deployment_parameters=nbase))
            opt=torch.optim.AdamW(list(model.parameters())+list(head.parameters()),lr=C['learning_rate'],weight_decay=C['weight_decay'])
            best=np.inf;stale=0;start=time.time();unstable=False
            for epoch,idx in enumerate(schedule[:C['epochs']],1):
                loader=DataLoader(Subset(train,idx.tolist()),batch_size=C['batch_size'],shuffle=False,num_workers=0,pin_memory=True)
                model.train();head.train();stats=np.zeros(7);n=0
                for b in loader:
                    b=to_device(b,device);opt.zero_grad(set_to_none=True)
                    pred,z0,roll=model(b,'true',True)
                    forecast=original_loss_fn(pred,b)
                    grounding,gpred=aux_loss(head,z0,roll,b,kind)
                    loss=forecast+lam*grounding
                    if not torch.isfinite(loss):unstable=True;break
                    loss.backward()
                    grad=float(torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(head.parameters()),1))
                    opt.step()
                    bs=len(pred);n+=bs
                    stats+=bs*np.asarray([float(forecast.detach()),float(grounding.detach()),grad,
                                           float(z0.detach().norm(dim=-1).mean()),
                                           float((roll[:,-1]-z0).detach().norm(dim=-1).mean()),
                                           float(gpred.detach().std()),float(gpred.detach().mean())])
                if unstable or not n:break
                val_bis,val_map=validate(model,vloader,store,device)
                d=dict(model=kind,lambda_weight=lam,epoch=epoch,train_forecast_loss=stats[0]/n,
                       train_grounding_loss=stats[1]/n,gradient_norm=stats[2]/n,
                       latent_norm=stats[3]/n,transition_latent_norm=stats[4]/n,
                       head_output_std=stats[5]/n,head_output_mean=stats[6]/n,
                       val_bis_mae=float(val_bis),val_map_mae=float(val_map),seconds=time.time()-start)
                rows.append(d);pd.DataFrame(rows).to_csv(out/'training_runs.csv',index=False)
                print(json.dumps(d),flush=True)
                if val_bis<best-1e-4:
                    best=float(val_bis);stale=0
                    torch.save(dict(model=model.state_dict(),head=head.state_dict(),kind=kind,lambda_weight=lam,
                                    epoch=epoch,val_bis_mae=best,val_map_mae=float(val_map)),
                               ROOT/'checkpoints'/f'{kind}_{lam:g}.pt')
                else:stale+=1
                if stale>=C['patience']:break
            summary.append(dict(model=kind,lambda_weight=lam,best_val_bis_mae=best,epochs_run=epoch,
                                unstable=unstable,seconds=time.time()-start))
            pd.DataFrame(summary).to_csv(out/'lambda_selection.csv',index=False)
    counts.insert(0,dict(model='baseline',lambda_weight=0,backbone_parameters=nbase,training_head_parameters=0,deployment_parameters=nbase))
    pd.DataFrame(counts).to_csv(out/'model_capacity.csv',index=False)
    selected={}
    for kind in ('state','transition'):
        valid=[x for x in summary if x['model']==kind and not x['unstable'] and np.isfinite(x['best_val_bis_mae'])]
        if not valid:raise RuntimeError(f'No stable {kind} run')
        best=min(x['best_val_bis_mae'] for x in valid)
        chosen=min((x for x in valid if x['best_val_bis_mae']<=best+.01),key=lambda x:x['lambda_weight'])
        selected[kind]=chosen['lambda_weight']
        shutil.copyfile(ROOT/'checkpoints'/f"{kind}_{chosen['lambda_weight']:g}.pt",ROOT/'checkpoints'/f'{kind}_selected.pt')
    (out/'selected_lambda.json').write_text(json.dumps(selected,indent=2))
    print('TRAIN_COMPLETE '+json.dumps(selected),flush=True)
if __name__=='__main__':main()
