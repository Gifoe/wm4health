"""Small frozen-input MLP probe for physiological change; no RSSM gradients."""
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.preprocessing import StandardScaler
from common import *
from probes import feature,target,VARS

class Probe(nn.Module):
    def __init__(self,dim):
        super().__init__();self.net=nn.Sequential(nn.Linear(dim,32),nn.GELU(),nn.Linear(32,3))
    def forward(self,x):return self.net(x)

@torch.no_grad()
def val_loss(model,x,y,m,device):
    model.eval();total=np.zeros(3);count=np.zeros(3)
    for xb,yb,mb in DataLoader(TensorDataset(x,y,m),batch_size=1024):
        p=model(xb.to(device)).cpu();sq=(p-yb).square()*mb
        total+=sq.sum(0).numpy();count+=mb.sum(0).numpy()
    return float(np.mean(total/np.maximum(count,1)))

def main():
    seed_all(0);device='cuda' if torch.cuda.is_available() else 'cpu'
    rows=[];history=[];(OUT/'nonlinear_checkpoints').mkdir(exist_ok=True)
    for k,h in enumerate(HORIZONS):
        ys={s:np.stack([target(var,k,'change')[0][s] for var in VARS],1) for s in ['train','val','test']}
        masks={s:np.stack([target(var,k,'change')[1][s] for var in VARS],1) for s in ys}
        scale=np.asarray([np.nanstd(ys['train'][masks['train'][:,j],j]) for j in range(3)],np.float32)
        assert np.all(np.isfinite(scale)) and np.all(scale>0)
        means=np.asarray([np.nanmean(ys['train'][masks['train'][:,j],j]) for j in range(3)],np.float32)
        for rep in ['dz_pred','dz_true']:
            x=feature(rep,k);scaler=StandardScaler().fit(x['train']);x={s:scaler.transform(v).astype(np.float32) for s,v in x.items()}
            rng=np.random.default_rng(0);available=np.flatnonzero(masks['train'].any(1))
            idx=rng.choice(available,size=min(CFG['nonlinear_train_windows'],len(available)),replace=False)
            def tensors(s,indices=None):
                xx=x[s] if indices is None else x[s][indices]
                yy=ys[s] if indices is None else ys[s][indices]
                mm=masks[s] if indices is None else masks[s][indices]
                yy=np.nan_to_num((yy-means)/scale).astype(np.float32)
                return (torch.from_numpy(xx),torch.from_numpy(yy),torch.from_numpy(mm.astype(np.float32)))
            train=tensors('train',idx);val=tensors('val')
            torch.manual_seed(0);model=Probe(x['train'].shape[1]).to(device)
            opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
            best=np.inf;stale=0;best_epoch=0
            for epoch in range(1,CFG['nonlinear_max_epochs']+1):
                model.train();losses=[]
                for xb,yb,mb in DataLoader(TensorDataset(*train),batch_size=512,shuffle=True):
                    xb=xb.to(device);yb=yb.to(device);mb=mb.to(device)
                    opt.zero_grad(set_to_none=True);p=model(xb)
                    loss=((p-yb).square()*mb).sum(0).div(mb.sum(0).clamp_min(1)).mean()
                    loss.backward();opt.step();losses.append(float(loss.detach()))
                vl=val_loss(model,*val,device)
                history.append({'representation':rep,'horizon_seconds':h*10,'epoch':epoch,
                                'train_loss':float(np.mean(losses)),'val_loss':vl})
                if vl<best-1e-4:
                    best=vl;stale=0;best_epoch=epoch
                    torch.save({'model':model.state_dict(),'input_mean':scaler.mean_,'input_scale':scaler.scale_,
                                'target_mean':means,'target_scale':scale,'epoch':epoch,'val_loss':best},
                               OUT/'nonlinear_checkpoints'/f'{rep}_{h}.pt')
                else:stale+=1
                if stale>=CFG['nonlinear_patience']:break
            ck=torch.load(OUT/'nonlinear_checkpoints'/f'{rep}_{h}.pt',map_location=device,weights_only=False)
            model.load_state_dict(ck['model']);model.eval();p=[]
            with torch.no_grad():
                for start in range(0,len(x['test']),2048):
                    p.append(model(torch.from_numpy(x['test'][start:start+2048]).to(device)).cpu().numpy())
            pred=np.concatenate(p)*scale+means
            np.save(ARRAYS/f'nonlinear_{rep}_{h}.npy',pred.astype(np.float32))
            for j,var in enumerate(VARS):
                met=weighted_metrics(ys['test'][:,j],pred[:,j],np.asarray(load('test','subject')),masks['test'][:,j])
                rows.append({'representation':rep,'target':var,'horizon_seconds':h*10,
                             'parameters':sum(p.numel() for p in model.parameters()),
                             'training_windows':len(idx),'best_epoch':best_epoch,'val_loss':best,**met})
            print('NONLINEAR',rep,h,best_epoch,flush=True)
    pd.DataFrame(rows).to_csv(OUT/'nonlinear_probe_metrics.csv',index=False)
    pd.DataFrame(history).to_csv(OUT/'nonlinear_training_history.csv',index=False)
    print('NONLINEAR_COMPLETE',flush=True)

if __name__=='__main__':main()
