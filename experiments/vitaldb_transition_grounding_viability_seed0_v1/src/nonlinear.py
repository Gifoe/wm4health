"""Round-6 32-unit frozen-input nonlinear probe for all three model rollouts."""
import numpy as np,pandas as pd,torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.preprocessing import StandardScaler
from evaluate import ROOT,OUT,ARR,AUDIT,H,VARS,feat,target,old,seed_all,weighted_metrics

class Probe(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(64,32),nn.GELU(),nn.Linear(32,3))
    def forward(self,x):return self.net(x)

@torch.no_grad()
def validation(model,tensors,device):
    model.eval();s=np.zeros(3);n=np.zeros(3)
    for x,y,m in DataLoader(TensorDataset(*tensors),batch_size=1024):
        q=model(x.to(device)).cpu();s+=(((q-y)**2)*m).sum(0).numpy();n+=m.sum(0).numpy()
    return float(np.mean(s/np.maximum(n,1)))

def main():
    seed_all(0);device='cuda' if torch.cuda.is_available() else 'cpu';rows=[];hist=[]
    baseline=pd.read_csv(AUDIT/'outputs/nonlinear_probe_metrics.csv')
    rows.extend(baseline[baseline.representation=='dz_pred'].assign(model='Baseline').to_dict('records'))
    for name in ('State-Grounded','Transition-Grounded'):
        for k,h in enumerate(H):
            xs={s:np.asarray(feat(name,s,'dz'))[:,k] for s in ('train','val','test')}
            ys={s:np.stack([target(s,var,k)[0] for var in VARS],1) for s in xs}
            masks={s:np.stack([target(s,var,k)[1] for var in VARS],1) for s in xs}
            means=np.asarray([np.nanmean(ys['train'][masks['train'][:,j],j]) for j in range(3)],np.float32)
            scales=np.asarray([np.nanstd(ys['train'][masks['train'][:,j],j]) for j in range(3)],np.float32)
            scaler=StandardScaler().fit(xs['train']);xs={s:scaler.transform(x).astype(np.float32) for s,x in xs.items()}
            def tensors(s,ix=None):
                x=xs[s] if ix is None else xs[s][ix];y=ys[s] if ix is None else ys[s][ix];m=masks[s] if ix is None else masks[s][ix]
                return torch.from_numpy(x),torch.from_numpy(np.nan_to_num((y-means)/scales).astype(np.float32)),torch.from_numpy(m.astype(np.float32))
            rng=np.random.default_rng(0);available=np.flatnonzero(masks['train'].any(1));ix=rng.choice(available,size=min(60000,len(available)),replace=False)
            train=tensors('train',ix);val=tensors('val');torch.manual_seed(0)
            model=Probe().to(device);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
            best=np.inf;stale=0;best_state=None;best_epoch=0
            for epoch in range(1,19):
                model.train();losses=[]
                for x,y,m in DataLoader(TensorDataset(*train),batch_size=512,shuffle=True):
                    x=x.to(device);y=y.to(device);m=m.to(device);opt.zero_grad(set_to_none=True)
                    q=model(x);loss=((q-y).square()*m).sum(0).div(m.sum(0).clamp_min(1)).mean()
                    loss.backward();opt.step();losses.append(float(loss.detach()))
                score=validation(model,val,device);hist.append(dict(model=name,horizon_seconds=h*10,epoch=epoch,train_loss=np.mean(losses),val_loss=score))
                if score<best-1e-4:
                    best=score;stale=0;best_epoch=epoch;best_state={key:v.detach().cpu().clone() for key,v in model.state_dict().items()}
                else:stale+=1
                if stale>=3:break
            model.load_state_dict(best_state);model.eval();p=[]
            with torch.no_grad():
                for start in range(0,len(xs['test']),2048):p.append(model(torch.from_numpy(xs['test'][start:start+2048]).to(device)).cpu().numpy())
            pred=np.concatenate(p)*scales+means
            for j,var in enumerate(VARS):
                met=weighted_metrics(ys['test'][:,j],pred[:,j],np.asarray(old('test','subject')),masks['test'][:,j])
                rows.append(dict(model=name,target=var,horizon_seconds=h*10,parameters=sum(v.numel() for v in model.parameters()),
                                 training_windows=len(ix),best_epoch=best_epoch,val_loss=best,**met))
            print('NONLINEAR',name,h,best_epoch,flush=True)
    pd.DataFrame(rows).to_csv(OUT/'nonlinear_probe_metrics.csv',index=False)
    pd.DataFrame(hist).to_csv(OUT/'nonlinear_training_history.csv',index=False)
    print('NONLINEAR_COMPLETE',flush=True)
if __name__=='__main__':main()
