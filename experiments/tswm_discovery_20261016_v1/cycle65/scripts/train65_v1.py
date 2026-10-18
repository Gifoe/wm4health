from models65 import *
import time
q=read(OUT/'QUALIFICATION65_V1.json');assert q['status']=='PASS';source_check();device=q['device'];torch.set_num_threads(4);pp=paths();fits=[];corpora=[];campaign=time.perf_counter()
def validation(model,raw,mean,std):
 model.eval();x=torch.tensor((raw[:256]-mean)/std,device=device);components={}
 with torch.no_grad(),torch.random.fork_rng(devices=[0] if device=='cuda' else []):
  for kind in ['block','prefix']:
   torch.manual_seed(650003);m=mask(x,kind,np.random.default_rng(650004));ss=side(model,x,m);components[kind]=float(model.calc_loss_valid(x,m,torch.ones_like(m),ss,0))
 model.train();return {**components,'mean':sum(components.values())/2}
for mechanism in MECHANISMS:
 for seed in SEEDS:
  key=f'{mechanism}_s{seed}';train=generate(16384,mechanism,seed,0);valid=generate(4096,mechanism,seed,1);test=generate(8192,mechanism,seed,2);datafile=pp['data_root']/(key+'.npz');np.savez_compressed(datafile,train=train,validation=valid,observational_test=test);mean=train.mean((0,2),keepdims=True);std=train.std((0,2),keepdims=True);corpora.append(dict(key=key,mechanism=mechanism,seed=seed,path=str(datafile),sha256=sha(datafile),training_records=len(train),validation_records=len(valid),reserved_observational_test_records=len(test)))
  X=torch.tensor((train-mean)/std,device=device)
  for regime in ['random','coverage']:
   torch.manual_seed(seed);np.random.seed(seed);model=csdi(device);opt=torch.optim.Adam(model.parameters(),lr=.001);ckpts=[];start=time.perf_counter();targets=0.
   for step in range(1,8001):
    model.train();ids=torch.randint(len(X),(64,),device=device);x=X[ids];draw=np.random.random() if regime=='coverage' else 0.
    m=model.get_randmask(torch.ones_like(x)) if draw<.5 else mask(x,'block' if draw<.75 else 'prefix');ss=side(model,x,m);opt.zero_grad();loss=model.calc_loss(x,m,torch.ones_like(m),ss,1);assert torch.isfinite(loss);loss.backward();opt.step();targets+=float((1-m).sum())
    if step%500==0:state('train_state65_v1.json',dict(status='TRAINING_NATIVE_CSDI',mechanism=mechanism,seed=seed,regime=regime,step=step,loss=float(loss),closed_fits=len(fits),device=device));print(json.dumps(dict(stage='CSDI',key=key,regime=regime,step=step,loss=float(loss))),flush=True)
    if step in [2000,8000]:
     vv=validation(model,valid,mean,std);p=pp['checkpoint_root']/f'{key}_csdi_{regime}_step{step}.pt';torch.save(dict(state_dict=model.state_dict(),optimizer=opt.state_dict(),config=config(),step=step,normalization_mean=mean,normalization_std=std,mechanism=mechanism,seed=seed,regime=regime,validation=vv,native_source_receipt_sha256=sha(OUT/'SOURCES65_V1.json')),p);ckpts.append(dict(path=str(p),sha256=sha(p),step=step,validation=vv));state('checkpoint_state65_v1.json',dict(status='SEALED',key=key,regime=regime,step=step,checkpoint_sha256=ckpts[-1]['sha256']))
   selected=min(ckpts,key=lambda z:z['validation']['mean']);fits.append(dict(key=key,model='CSDI_native_'+regime,mechanism=mechanism,seed=seed,parameters=sum(v.numel() for v in model.parameters()),optimizer_updates=8000,trajectory_presentations=8000*64,target_scalar_exposures=targets,elapsed_seconds=time.perf_counter()-start,checkpoints=ckpts,selected_checkpoint=selected,data_sha256=corpora[-1]['sha256'],selection='fixed block/prefix validation DSM only'))
   state('fit_registry65_v1.json',dict(status='PARTIAL',fits=fits,corpora=corpora));del model,opt
  tx,ty=transitions(train);vx,vy=transitions(valid);tx=torch.tensor(tx,device=device);ty=torch.tensor(ty,device=device);vx=torch.tensor(vx,device=device);vy=torch.tensor(vy,device=device);torch.manual_seed(seed);model=MDN().to(device);opt=torch.optim.Adam(model.parameters(),lr=.001);ckpts=[];start=time.perf_counter()
  for step in range(1,8001):
   ids=torch.randint(len(tx),(512,),device=device);opt.zero_grad();loss=model.nll(tx[ids],ty[ids]);assert torch.isfinite(loss);loss.backward();opt.step()
   if step%1000==0:state('train_state65_v1.json',dict(status='TRAINING_CAUSAL_MDN',mechanism=mechanism,seed=seed,step=step,loss=float(loss),closed_fits=len(fits),device=device))
   if step in [2000,8000]:
    model.eval()
    with torch.no_grad():nll=float(model.nll(vx,vy))
    p=pp['checkpoint_root']/f'{key}_MDN_step{step}.pt';torch.save(dict(state_dict=model.state_dict(),optimizer=opt.state_dict(),step=step,validation_NLL=nll,mechanism=mechanism,seed=seed),p);ckpts.append(dict(path=str(p),sha256=sha(p),step=step,validation_NLL=nll));model.train()
  fits.append(dict(key=key,model='causal_MDN',mechanism=mechanism,seed=seed,parameters=sum(v.numel() for v in model.parameters()),optimizer_updates=8000,target_scalar_exposures=8000*512,elapsed_seconds=time.perf_counter()-start,checkpoints=ckpts,selected_checkpoint=min(ckpts,key=lambda z:z['validation_NLL']),data_sha256=corpora[-1]['sha256'],selection='validation one-step conditional NLL only'));state('fit_registry65_v1.json',dict(status='PARTIAL',fits=fits,corpora=corpora))
dump(OUT/'FITS65_V1.json',dict(utc=utc(),status='ALL_18_MODELS_FROZEN',fits=fits,corpora=corpora,unique_training_corpora=6,models=len(fits),optimizer_updates=sum(r['optimizer_updates'] for r in fits),device=device,config=config(),elapsed_seconds=time.perf_counter()-campaign,protocol_sha256=sha(ROOT/'PROTOCOL65_V1.md'),accepted_final_directions=0));state('train_state65_v1.json',dict(status='ALL_18_MODELS_FROZEN',models=len(fits),optimizer_updates=sum(r['optimizer_updates'] for r in fits)));print(json.dumps(dict(status='ALL_18_MODELS_FROZEN',models=len(fits))),flush=True)
