"""Native API/noise and AR cache checks; synthetic preflight only, no VAL scores."""
from common40 import *
import datetime
O=R/'outputs';assert not (O/'qualification.json').exists()
seed(202640);f=make_fnp();a=ARMixture().cuda();f.eval();a.eval()
seq=torch.randn(2,288,1,device=DEVICE);idx=torch.arange(288,device=DEVICE)[None].repeat(2,1);c=coordinates(idx,torch.full((2,1),192,device=DEVICE));checks=[]
before=statehash(f)
for r in REVEALS:
 batch=fnp_batch(seq,c,r)
 # Small source API fixtures at all reveal counts; keep full public target layout.
 f.timesteps=2;seed(700+r);native=f.predict(batch.xc,batch.yc,batch.xt,num_samples=2).loc
 seed(700+r);replay=replay_fnp(f,batch.xc,batch.yc,batch.xt,2)
 err=float((native-replay).abs().max());assert err<2e-6,err
 checks.append({'r':r,'native_sampler_max_error':err})
 f.train();seed(800+r);native_loss=f(batch).loss
 seed(800+r);y0=torch.randn_like(batch.yt);t=torch.rand((2,batch.yt.shape[1],1),device=DEVICE)
 z=t*batch.yt+(1-t)*y0;manual=F.mse_loss(f.predict_velocity(z,t,batch),batch.yt-y0)
 err=float((native_loss-manual).abs());assert err<1e-6,err;checks[-1]['native_training_loss_error']=err
assert statehash(f)==before
cache=None
for r in REVEALS:
 cache=a.context(seq,c,r,cache);fresh=a.context(seq,c,r)
 err=float((cache[2]-fresh[2]).abs().max());assert err<2e-5,err
 seed(900+r);x=a.sample_cached(seq,c,r,cache,4)
 seed(900+r);z=a.sample_cached(seq,c,r,fresh,4)
 err2=float((x-z).abs().max());assert err2<2e-4,err2
 checks.append({'r':r,'cache_hidden_error':err,'cache_sample_error':err2})
# Fixed synthetic input; excludes real validation and independent TEST.
seq=torch.randn(32,288,1,device=DEVICE);idx=torch.arange(288,device=DEVICE)[None].repeat(32,1);c=coordinates(idx,torch.full((32,1),192,device=DEVICE));performance={}
for name,model in [('flownp',f),('ar8mix',a)]:
 model.train();opt=torch.optim.AdamW(model.parameters(),lr=3e-4);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
 for step in range(100):
  opt.zero_grad(set_to_none=True);loss=model(fnp_batch(seq,c,REVEALS[step%3])).loss if name=='flownp' else model.loss(seq,c)
  assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
 torch.cuda.synchronize();performance[name]={'100_training_steps_seconds':time.perf_counter()-start,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,'parameters':sum(p.numel() for p in model.parameters())}
assert max(x['peak_allocated_GiB'] for x in performance.values())<16
out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checks':checks,'preflight':performance,'native_model_not_mutated_during_api_checks':True,'synthetic_preflight_models_discarded':True,'validation_scored':False,'test_scored':False}
(O/'qualification.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))
