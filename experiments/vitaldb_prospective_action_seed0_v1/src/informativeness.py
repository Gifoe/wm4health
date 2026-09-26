"""Matched-capacity direct supervised control for future-action informativeness."""
import json,time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader,Subset
from core import *
from train import loss_fn

OUT=ROOT/'outputs'

class DirectPredictor(nn.Module):
    def __init__(self):
        super().__init__();d=CFG['model_dim']
        self.encoder=nn.GRU(14,d,2,batch_first=True,dropout=.1)
        self.latent=nn.LayerNorm(d)
        self.future=nn.Sequential(nn.Linear(F*4,d),nn.GELU())
        self.head=nn.Sequential(nn.Linear(2*d,d),nn.GELU(),nn.Linear(d,F*2))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)

    def forward(self,b,use_future):
        x=torch.cat([b['state'],b['action'],b['demo'][:,None,:].expand(-1,H,-1)],-1)
        _,h=self.encoder(x);z=self.latent(h[-1])
        f=b['future_action'].flatten(1) if use_future else torch.zeros_like(b['future_action']).flatten(1)
        return b['current'][:,None,:]+self.head(torch.cat([z,self.future(f)],-1)).view(-1,F,2)

@torch.no_grad()
def validate(m,loader,store,device,use_future):
    m.eval();summed=0.;count=0.;scale=float(store.norm['state_std'][0])
    for b in loader:
        b=to_device(b,device);p=m(b,use_future)
        e=(p[:,:,0]-b['target'][:,:,0]).abs()*b['target_mask'][:,:,0]
        summed+=float(e.sum());count+=float(b['target_mask'][:,:,0].sum())
    return summed/max(count,1)*scale

def main():
    seed_all();store=Store();train=Windows(store,'train');val=Windows(store,'val',CFG['validation_stride'])
    schedules=np.load(OUT/'epoch_train_indices.npy')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    val_loader=DataLoader(val,batch_size=512,num_workers=0,pin_memory=True)
    rows=[];summaries=[]
    for name,use_future in [('Direct history control',False),('Direct future control',True)]:
        seed_all();m=DirectPredictor().to(device)
        opt=torch.optim.AdamW(m.parameters(),lr=CFG['learning_rate'],weight_decay=CFG['weight_decay'])
        best=np.inf;stale=0;start=time.time();local=[]
        for epoch,idx in enumerate(schedules,1):
            loader=DataLoader(Subset(train,idx.tolist()),batch_size=CFG['batch_size'],shuffle=False,num_workers=0,pin_memory=True)
            m.train();total=0.;n=0
            for b in loader:
                b=to_device(b,device);opt.zero_grad(set_to_none=True)
                p=m(b,use_future);loss=loss_fn(p,b)
                if not torch.isfinite(loss):raise RuntimeError('nonfinite '+name)
                loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
                total+=float(loss.detach())*len(p);n+=len(p)
            score=validate(m,val_loader,store,device,use_future)
            row={'model':name,'epoch':epoch,'train_loss':total/n,'val_bis_mae':score,'seconds':time.time()-start}
            rows.append(row);local.append(row);pd.DataFrame(rows).to_csv(OUT/'informativeness_training_history.csv',index=False)
            print(json.dumps(row),flush=True)
            if score<best-1e-4:
                best=score;stale=0
                torch.save({'model':m.state_dict(),'use_future':use_future,'name':name,
                            'epoch':epoch,'parameters':sum(p.numel() for p in m.parameters())},
                           OUT/'checkpoints'/('direct_future.pt' if use_future else 'direct_history.pt'))
            else:stale+=1
            if stale>=CFG['patience']:break
        ck=torch.load(OUT/'checkpoints'/('direct_future.pt' if use_future else 'direct_history.pt'),map_location='cpu',weights_only=False)
        summaries.append({'model':name,'use_future':use_future,'parameters':ck['parameters'],
                          'best_epoch':ck['epoch'],'epochs_run':len(local),'best_val_bis_mae':best,
                          'seconds':time.time()-start})
    pd.DataFrame(summaries).to_csv(OUT/'informativeness_model_summary.csv',index=False)
    print('INFORMATIVENESS_TRAIN_COMPLETE',flush=True)

@torch.no_grad()
def predict(store,ds,checkpoint,use_future):
    device='cuda' if torch.cuda.is_available() else 'cpu'
    m=DirectPredictor().to(device);m.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=False)['model']);m.eval()
    output=[]
    for b in DataLoader(ds,batch_size=512,num_workers=0,pin_memory=True):
        b=to_device(b,device);output.append(m(b,use_future).cpu().numpy())
    return np.concatenate(output)*store.norm['state_std'][[0,2]]+store.norm['state_mean'][[0,2]]

def evaluate():
    from evaluate import abs_error,scalar_error,bootstrap,score_rows
    store=Store();ds=Windows(store,'test');ref=np.load(OUT/'test_reference.npz')
    subjects=ref['subject'];target=ref['target'];mask=ref['mask']
    subsets={'overall':np.ones(len(ds),bool),'large_transition':ref['large_label']>0,
             'high_future_divergence':ref['divergence']>=max(float(json.loads((OUT/'divergence_protocol.json').read_text())['train_quantile_boundaries'][-1]),.25)}
    for k,label in [(1,'initiation'),(2,'increase'),(3,'decrease'),(4,'stop')]:
        subsets[label]=ref['large_label']==k
    errors={};forecast=[]
    for key,name,future in [('history','Direct history control',False),('future','Direct future control',True)]:
        p=predict(store,ds,OUT/'checkpoints'/f'direct_{key}.pt',future)
        np.savez_compressed(OUT/f'prediction_direct_{key}.npz',prediction=p)
        errors[key]=abs_error(p,target,mask)
        forecast+=score_rows(name,'future' if future else 'history',errors[key],subjects,subsets)
        print('DIRECT_INFER',name,flush=True)
    pd.DataFrame(forecast).to_csv(OUT/'informativeness_forecast_metrics.csv',index=False)
    rows=[]
    for subset,keep in subsets.items():
        for h in [3,6,18,30,0]:
            delta=scalar_error(errors['history'],h)-scalar_error(errors['future'],h)
            point,lo,hi,n=bootstrap(delta[keep],subjects[keep])
            rows.append({'subset':subset,'horizon_seconds':h*10,'direct_future_value':point,
                         'ci_low':lo,'ci_high':hi,'n_windows':int(np.isfinite(delta[keep]).sum()),'n_patients':n})
    pd.DataFrame(rows).to_csv(OUT/'future_action_informativeness.csv',index=False)
    print('INFORMATIVENESS_EVAL_COMPLETE',flush=True)

if __name__=='__main__':
    import sys
    evaluate() if '--evaluate' in sys.argv else main()
