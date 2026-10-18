from contiguous_loader62_v1 import *
ad,models,compat=load_native();df,dp=load_data();seed(6202);util=importlib.import_module('ts_benchmark.baselines.utils')
tr=int(int(len(df)*.8)*.75);tv=int(len(df)*.8);a=adapter(ad,models,'Informer_PRO');a.multi_forecasting_hyper_param_tune(df.iloc[:tr]);a.config.task_name='short_term_forecast'
norm=(df-df.iloc[:tr].mean())/df.iloc[:tr].std(ddof=0);train,val=util.train_val_split(norm.iloc[:tv],.75,96)
ds,_=util.forecasting_data_provider(train,a.config,1,32,True,True);vs,_=util.forecasting_data_provider(val,a.config,1,32,True,False)
Base,old,new=install();rng=np.random.default_rng(6202);checked=0
for data in [ds,vs]:
 for i in rng.choice(len(data),size=64,replace=False):
  before=old(data,int(i));after=new(data,int(i));assert all(torch.equal(x,y) and x.dtype==y.dtype for x,y in zip(before,after));checked+=1
# Compare same initialized core and same RNG on a whole batch including gradients.
b0=tuple(torch.stack([old(ds,j)[k] for j in range(4)]) for k in range(4));b1=tuple(torch.stack([new(ds,j)[k] for j in range(4)]) for k in range(4))
m=models['Informer_PRO'](a.config).cuda();snap={k:v.detach().clone() for k,v in m.state_dict().items()};grads=[];outs=[]
for batch in [b0,b1]:
 m.load_state_dict(snap);m.zero_grad(set_to_none=True);seed(6203);x,y,xm,ym=[v.cuda() for v in batch];dec=torch.cat([y[:,:48],torch.zeros_like(y[:,-96:])],1);out=m(x,xm,dec,ym);loss=(out-y[:,-96:]).square().mean();loss.backward();outs.append(out.detach().cpu());grads.append({n:p.grad.detach().cpu().clone() for n,p in m.named_parameters() if p.grad is not None})
fo=float((outs[0]-outs[1]).abs().max());gg=max(float((grads[0][k]-grads[1][k]).abs().max()) for k in grads[0]);assert fo<2e-6 and gg<2e-6,(fo,gg)
# No optimization or model access: stress the alternative dataset copy path beyond failed runs.
t=__import__('time').perf_counter()
for j in range(180000):
 sample=new(ds,(j*137)%len(ds))
 if j%30000==0:print(json.dumps({'stress_windows':j,'finite':all(torch.isfinite(x).all().item() for x in sample)}),flush=True)
dump(OUT/'CONTIGUOUS_LOADER_QUALIFICATION62_V1.json',{'utc':utc(),'status':'PASS','bitwise_window_comparisons':checked,'forward_max_difference':fo,'gradient_max_difference':gg,'stress_windows':180000,'stress_seconds':__import__('time').perf_counter()-t,'optimizer_updates':0,'upstream_unchanged':True,'scope':'Explicit CPU ndarray copying only; does not establish root cause of native access violations.'})
print(json.dumps({'status':'PASS','bitwise_windows':checked,'forward':fo,'gradient':gg,'stress_windows':180000}))
