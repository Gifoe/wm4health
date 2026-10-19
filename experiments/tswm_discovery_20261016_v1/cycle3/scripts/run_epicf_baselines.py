"""Official single-policy baselines plus preregistered information ablations.

All additional ablations keep the exact native Transformer and training budget.
This intentionally reproduces the released train-as-validation protocol. It is
not a generalization-controlled method comparison. No test labels select weights.
"""
import contextlib,datetime,hashlib,json,pathlib,random,sys,time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,TensorDataset
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'sources'/'epi-cf-benchmark'/'src'))
from epicf_benchmark.models import FluTransformer
from epicf_benchmark.transformer import train_transformer,train_epoch,eval_epoch,predict
from epicf_benchmark.kde import train_kde,infer_kde,build_train_parser as kde_train_parser,build_infer_parser as kde_infer_parser
from epicf_benchmark.transformer import build_train_parser
from epicf_benchmark.evaluate import evaluate_cate_rmse
SEEDS=[41,42,43]
VARIANTS=['official','no_policy','no_states','no_policy_no_states']
def dump(path,obj):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,indent=2),encoding='utf-8')
def seed_all(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
def metrics(yf,ycf,pf,pcf):
 ranges=np.maximum(np.ptp(ycf,axis=1),1.0)
 return {'factual_rmse_zscore':float(np.sqrt(np.mean((pf-yf)**2))),'cf_rmse_zscore':float(np.sqrt(np.mean((pcf-ycf)**2))),'cf_nrmse_county_range_upstream':float(np.mean(np.sqrt(np.mean((pcf-ycf)**2,axis=1))/ranges)),'effect_rmse_zscore':float(np.sqrt(np.mean(((pf-pcf)-(yf-ycf))**2))),'cate_rmse_upstream':evaluate_cate_rmse(yf,ycf,pf,pcf,num_units=158)}
def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
 out=ROOT/'outputs'/'epicf_baselines';out.mkdir(parents=True,exist_ok=True);data=ROOT/'data'/'single_official'
 plan={'frozen_before_models_trained_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_revision':'8f7df27507dc8b01422bb373417c8d0d3efffae2','data_record':20762823,'setting':'single policy, all 158 source blocks including duplicates','seeds':SEEDS,'variants':VARIANTS,'native_architecture':'FluTransformer input18,dmodel256,nhead8,nlayers6,ff1024,dropout.15','budget':{'epochs':100,'lr':.001,'batch_size':8,'optimizer':'Adam'},'validation':'Native train-file/all-training-indices validation; explicitly not independent','ablation':'Zero inputs, train and evaluate with same fixed mask; no distribution-shift-only masking claim','primary_metrics':['cf_rmse_zscore','effect_rmse_zscore'],'secondary':['upstream per-county-range normalized metrics'],'prohibited_inference':'Scores do not establish prospective world-model fidelity or causal identification. No novelty claim from protocol defects.'}
 planpath=out/'plan.json'
 if planpath.exists():
  old=json.loads(planpath.read_text());assert all(old[k]==plan[k] for k in ['source_revision','data_record','seeds','variants','budget'])
 else:dump(planpath,plan)
 device=torch.device('cuda');assert torch.cuda.is_available()
 arrays={}
 for split in ['train','test']:
  y=pd.read_csv(data/f'y_{split}.csv',header=None).to_numpy().reshape(158,168)
  a=pd.read_csv(data/f'a_{split}.csv',header=None).to_numpy().reshape(158,168,1)
  x=pd.read_csv(data/f'x_{split}.csv').to_numpy().reshape(158,168,17)
  arrays[split]=(np.concatenate([a,x],axis=2).astype('float32'),y.astype('float32'))
 yf=arrays['train'][1];ycf=arrays['test'][1]
 rows=[]
 status=lambda **kw:dump(out/'status.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':len(rows),'total':13,**kw})
 # The small deterministic KDE uses the unmodified released trainer and inferencer.
 kdeout=out/'kde';kdeout.mkdir(exist_ok=True)
 if not (kdeout/'result.json').exists():
  with (kdeout/'train.log').open('w',encoding='utf-8') as logfile,contextlib.redirect_stdout(logfile):
   train_kde(kde_train_parser().parse_args(['--data-dir',str(data),'--output-dir',str(kdeout)]))
   for split in ['train','test']:
    infer_kde(kde_infer_parser().parse_args(['--data-dir',str(data),'--model-dir',str(kdeout),'--output-dir',str(kdeout),'--split',split,'--prefix',split]))
  pf=np.load(kdeout/'train.npy');pcf=np.load(kdeout/'test.npy');row={'variant':'kde_official','seed':None,**metrics(yf,ycf,pf,pcf)};dump(kdeout/'result.json',row)
 rows.append(json.loads((kdeout/'result.json').read_text()));status(state='running')
 for variant in VARIANTS:
  for seed in SEEDS:
   run=out/f'{variant}_seed{seed}';run.mkdir(exist_ok=True)
   if (run/'result.json').exists():rows.append(json.loads((run/'result.json').read_text()));continue
   seed_all(seed);start=time.perf_counter();status(state='training',variant=variant,seed=seed)
   model_path=run/'transformer.pt'
   with (run/'train.log').open('w',encoding='utf-8') as logfile,contextlib.redirect_stdout(logfile):
    if variant=='official':
     args=build_train_parser().parse_args(['--data-dir',str(data),'--output-dir',str(run),'--device','cuda'])
     train_transformer(args)
     model=FluTransformer(input_dim=18).to(device)
     model.load_state_dict(torch.load(model_path,map_location=device,weights_only=True))
     loaders={split:DataLoader(TensorDataset(torch.tensor(x),torch.tensor(y)),batch_size=8,shuffle=False) for split,(x,y) in arrays.items()}
    else:
     transformed={}
     for split,(x,y) in arrays.items():
      masked=x.copy()
      if 'no_policy' in variant:masked[:,:,0]=0
      if 'no_states' in variant:masked[:,:,14:18]=0
      transformed[split]=(masked,y)
     train_loader=DataLoader(TensorDataset(*[torch.tensor(v) for v in transformed['train']]),batch_size=8,shuffle=True)
     loaders={split:DataLoader(TensorDataset(*[torch.tensor(v) for v in pair]),batch_size=8,shuffle=False) for split,pair in transformed.items()}
     model=FluTransformer(input_dim=18).to(device);opt=torch.optim.Adam(model.parameters(),lr=.001);criterion=torch.nn.MSELoss();best=float('inf')
     # Match upstream's validation construction, evaluation frequency and budget.
     for epoch in range(1,101):
      loss=train_epoch(model,train_loader,opt,criterion,device)
      val=eval_epoch(model,loaders['train'],criterion,device)
      if val<=best:best=val;torch.save(model.state_dict(),model_path)
      print(json.dumps({'epoch':epoch,'train_loss':loss,'train_as_val_loss':val,'best_train_as_val_loss':best}),flush=True)
     model.load_state_dict(torch.load(model_path,map_location=device,weights_only=True))
    # Native trainer consumes two evaluation passes at the end. For the ablations
    # no inference labels flow into gradient or weight selection.
    pf=predict(model,loaders['train'],device);pcf=predict(model,loaders['test'],device)
   result={'variant':variant,'seed':seed,'seconds':time.perf_counter()-start,'parameters':sum(p.numel() for p in model.parameters()),'checkpoint_sha256':hashlib.sha256(model_path.read_bytes()).hexdigest(),**metrics(yf,ycf,pf,pcf)}
   np.savez_compressed(run/'predictions.npz',factual=pf,counterfactual=pcf)
   dump(run/'result.json',result);rows.append(result);dump(out/'results.json',rows);print(json.dumps(result),flush=True)
   del model;torch.cuda.empty_cache()
 dump(out/'results.json',rows);status(state='complete')
if __name__=='__main__':main()
