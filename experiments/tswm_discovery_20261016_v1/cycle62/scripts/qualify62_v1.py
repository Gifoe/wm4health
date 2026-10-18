from native62 import *
import pandas as pd,platform
from types import SimpleNamespace
rows=verify_sources();ad,models,compat=load_native();df,dp=load_data();seed(6201)
tv=int(len(df)*.8);tr=int(tv*.75);train=df.iloc[:tr]
util=importlib.import_module('ts_benchmark.baselines.utils');results=[]
for arm in models:
 a=adapter(ad,models,arm);a.multi_forecasting_hyper_param_tune(train);a.config.task_name='short_term_forecast';m=models[arm](a.config).cuda()
 norm=(train-train.mean())/train.std(ddof=0)
 ds,dl=util.forecasting_data_provider(norm,a.config,timeenc=1,batch_size=4,shuffle=False,drop_last=False)
 x,y,xm,ym=next(iter(dl));assert torch.equal(x[:,-a.config.label_len:,:],y[:,:a.config.label_len,:])
 def dec(v):return torch.cat([v[:,:a.config.label_len],torch.zeros_like(v[:,-96:])],dim=1)
 yy=y.clone();yy[:,-96:]=123456.;assert torch.equal(dec(y),dec(yy))
 pred=m(x.cuda(),xm.cuda(),dec(y).cuda(),ym.cuda());assert pred.shape==(4,96,7) and torch.isfinite(pred).all()
 loss=(pred-y[:,-96:].cuda()).square().mean();loss.backward();assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
 results.append({'arm':arm,'shape':list(pred.shape),'parameters':sum(p.numel() for p in m.parameters()),'finite_backward':True,'decoder_future_target_sentinel_difference':0.,'loss_training_only_probe':loss.item()})
 del m;torch.cuda.empty_cache()
dump(OUT/'QUALIFICATION62_V1.json',{'utc':utc(),'status':'PASS','source_revision':'733d986d349e9e71ded5d9cdeeeed8d3323d83b4','source_files':rows,'imported_sources':imported_sources(),'data_sha256':sha(dp),'data_git_blob':blob(dp.read_bytes()),'shape':list(df.shape),'tv_boundary':tv,'train_boundary':tr,'protocol_sha256':sha(ROOT/'PROTOCOL62_V1.md'),'compatibility':compat,'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),'arms':results})
print(json.dumps({'status':'PASS','arms':results,'formal_optimizer_updates':0}),flush=True)
