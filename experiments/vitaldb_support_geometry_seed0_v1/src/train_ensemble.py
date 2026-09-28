"""Train prospective RSSMs with the unmodified Round-2 objective and schedule protocol."""
import json,shutil,time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,Subset
from common import *
from train import loss_fn,validate

def main():
    seed_all(0)
    OUT.mkdir(exist_ok=True);(OUT/'checkpoints').mkdir(exist_ok=True)
    store=Store();train=Windows(store,'train');val=Windows(store,'val',CFG['validation_stride'])
    device='cuda' if torch.cuda.is_available() else 'cpu'
    vl=DataLoader(val,batch_size=512,num_workers=0,pin_memory=True)
    rows=[];hist=[]
    source_ck=SOURCE/'outputs/checkpoints/true.pt'
    source_summary=pd.read_csv(SOURCE/'outputs/model_summary.csv')
    sr=source_summary[source_summary.policy=='true'].iloc[0]
    shutil.copy2(source_ck,OUT/'checkpoints/seed0.pt')
    ck=torch.load(source_ck,map_location='cpu',weights_only=False)
    assert ck['epoch']==int(sr.best_epoch) and ck['parameters']==58946
    assert len(train)==370566 and len(val)==13263
    rows.append(dict(seed=0,reused_round2=True,parameters=ck['parameters'],best_epoch=ck['epoch'],
                     epochs_run=int(sr.epochs_run),best_val_bis_mae=float(ck['val_bis_mae']),
                     train_windows=len(train),windows_per_epoch=CFG['training_windows_per_epoch'],seconds=float(sr.seconds)))
    for seed in CFG['seeds'][1:]:
        seed_all(seed);start=time.time()
        rng=np.random.default_rng(seed)
        schedules=[rng.permutation(len(train))[:CFG['training_windows_per_epoch']].copy()
                   for _ in range(CFG['epochs'])]
        model=RSSM().to(device);model.set_action_stats(store)
        opt=torch.optim.AdamW(model.parameters(),lr=CFG['learning_rate'],weight_decay=CFG['weight_decay'])
        best=np.inf;stale=0
        for epoch,idx in enumerate(schedules,1):
            loader=DataLoader(Subset(train,idx.tolist()),batch_size=CFG['batch_size'],shuffle=False,
                              num_workers=0,pin_memory=True)
            model.train();total=0.;n=0
            for b in loader:
                b=to_device(b,device);opt.zero_grad(set_to_none=True)
                pred,_,_=model(b,'true');loss=loss_fn(pred,b)
                if not torch.isfinite(loss):raise RuntimeError(f'seed{seed} nonfinite loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
                total+=float(loss.detach())*len(pred);n+=len(pred)
            scores=validate(model,vl,store,device,'true')
            rec=dict(seed=seed,epoch=epoch,train_loss=total/n,val_bis_mae=float(scores[0]),
                     val_map_mae=float(scores[1]),seconds=time.time()-start)
            hist.append(rec);pd.DataFrame(hist).to_csv(OUT/'ensemble_training_history.csv',index=False)
            print(json.dumps(rec),flush=True)
            if scores[0]<best-1e-4:
                best=float(scores[0]);stale=0
                torch.save({'model':model.state_dict(),'policy':'true','name':'Prospective RSSM',
                            'seed':seed,'epoch':epoch,'parameters':sum(p.numel() for p in model.parameters()),
                            'val_bis_mae':best},OUT/'checkpoints'/f'seed{seed}.pt')
            else:stale+=1
            if stale>=CFG['patience']:break
        rows.append(dict(seed=seed,reused_round2=False,parameters=sum(p.numel() for p in model.parameters()),
                         best_epoch=torch.load(OUT/'checkpoints'/f'seed{seed}.pt',weights_only=False)['epoch'],
                         epochs_run=epoch,best_val_bis_mae=best,train_windows=len(train),
                         windows_per_epoch=CFG['training_windows_per_epoch'],seconds=time.time()-start))
        pd.DataFrame(rows).to_csv(OUT/'ensemble_model_summary.csv',index=False)
    pd.DataFrame(rows).to_csv(OUT/'ensemble_model_summary.csv',index=False)
    jwrite('data_summary.json',{'source_data_summary':json.loads((SOURCE/'outputs/data_summary.json').read_text()),
      'seeds':CFG['seeds'],'seed0_reused_exact_round2_true_checkpoint':True,
      'windows':{s:len(Windows(store,s)) for s in ['train','val','test']},
      'n_cases':len(store.cases),'n_subjects':len({int(c['subjectid']) for c in store.cases.values()}),
      'target':'BIS','secondary_target':'MAP','sample_seconds':10,'history_steps':180,'horizon_steps':30})
    print('ENSEMBLE_TRAIN_COMPLETE',flush=True)

if __name__=='__main__':main()
