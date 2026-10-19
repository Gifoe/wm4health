"""Source-faithful full COFE model/loss on ETTh1, plus implementation controls.

Disclosed50x100update/24->24 configuration, manual Torch orchestration. Training
origins are IID uniform on native sampler support rather than a GluonTS iterable;
this is explicitly CLOSE source/config replication, not a paper-table MATCHED row.
"""
import datetime,hashlib,json,math,os,pathlib,time,types
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal
from native_cofe import ROOT,build
def dump(p,o):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(o,indent=2),encoding='utf-8')
def fixed_decode(self,x_rec,x_pred):
 B,_,_=x_rec.shape
 rec=self.decoder_mu(x_rec).reshape(B,self.freq,self.patch_len,self.enc_in).reshape(B,-1,self.enc_in)[:,:self.input_len]
 mu=self.decoder_mu(x_pred).reshape(B,self.step,self.patch_len,self.enc_in).reshape(B,-1,self.enc_in)[:,:self.pred_len]
 sigma_logits=self.decoder_sigma(x_pred).reshape(B,self.step,self.patch_len,self.enc_in).reshape(B,-1,self.enc_in)[:,:self.pred_len]
 rec=self.revin(rec,'denorm');mu=self.revin(mu,'denorm')
 scale=F.softplus(sigma_logits)*self.revin.stdev/(self.revin.affine_weight.abs()+1e-10)+1e-6
 return rec,Normal(mu,scale)
def getbatch(data,origins):
 past=data[origins[:,None]+torch.arange(-24,0,device='cuda')[None,:]]
 future=data[origins[:,None]+torch.arange(24,device='cuda')[None,:]]
 return SimpleNamespace(past_target_cdf=past,future_target_cdf=future)
def score(model,data,origins,mean,std,seed,offset=0.):
 model.eval();torch.manual_seed(seed)
 batch=getbatch(data,origins);batch.past_target_cdf=batch.past_target_cdf+offset
 target=batch.future_target_cdf+offset
 with torch.no_grad():sample=model.forecast(batch,num_samples=100)
 assert torch.isfinite(sample).all()
 qs=torch.arange(1,20,device='cuda',dtype=sample.dtype)/20
 quantile=torch.quantile(sample,qs,dim=1)
 target_raw=target*std+mean;qr=quantile*std+mean
 # Native per-sequence,19-quantile normalized quantile score on original units.
 residual=target_raw[None]-qr
 loss=2*torch.maximum(qs[:,None,None,None]*residual,(qs[:,None,None,None]-1)*residual)
 native=(loss.sum((2,3))/target_raw.abs().sum((1,2))[None]).mean()
 err=sample.mean(1)-target
 low=torch.quantile(sample,.05,dim=1);high=torch.quantile(sample,.95,dim=1)
 ordered=sample.sort(dim=1).values;s=sample.shape[1];weights=(2*torch.arange(1,s+1,device='cuda')-s-1).to(sample.dtype)
 empirical=(sample-target[:,None]).abs().mean(1)-(ordered*weights[None,:,None,None]).sum(1)/(s*s)
 median=torch.quantile(sample,.5,dim=1)
 nd=((median*std+mean-target_raw).abs().sum((1,2))/target_raw.abs().sum((1,2))).mean()
 row={'native_raw_CRPS_quantile19':native.item(),'native_raw_ND':nd.item(),'normalized_MSE':err.square().mean().item(),'normalized_MAE':(median-target).abs().mean().item(),'empirical_CRPS_normalized':empirical.mean().item(),'coverage90':((target>=low)&(target<=high)).float().mean().item(),'interval90_width_normalized':(high-low).mean().item(),'windows':len(origins)}
 return row,sample.cpu().numpy(),target.cpu().numpy()
def main():
 torch.set_num_threads(int(os.environ.get('TSWM_CPU_THREADS','4')));torch.backends.cuda.matmul.allow_tf32=False
 tag=os.environ.get('TSWM_COFE_OUT','cofe_etth1');assert '/' not in tag and '\\' not in tag and tag not in ['.','..']
 out=ROOT/'outputs'/tag;out.mkdir(parents=True,exist_ok=True)
 plan={'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'CLOSE source/config replication plus isolated implementation controls; not paper-table MATCHED. Full638917parameter model and unchanged native loss,24context,24horizon.','source_revision':'a53409aa6d97c65b5b36b7f6b27fc5fbdb5e3df1','dataset_sha256':hashlib.sha256((ROOT/'data'/'ETTh1.csv').read_bytes()).hexdigest(),'split':{'train_end':8640,'val_end':11520,'test_end':14400,'rolling_stride':96,'eval_windows_per_split':30,'train_origin_support':[192,8616]},'scaler':'Training-only per-channel mean and unbiased standard deviation, matching author StandardScaler','training':{'seed':1,'epochs':50,'updates_per_epoch':100,'batch':48,'optimizer':'Adam','lr':.001,'gradient_clip_norm':.5,'checkpoint':'minimum raw-unit native19-quantile CRPS on validation only; no test selection'},'declared_sampler_difference':'Uniform IID train origins on native min_past192/min_future24 support, versus GluonTS ExpectedNumInstanceSampler iterable. Worker count0; native model/attention/scaler/loss bodies retained by selective loading.','variants':['released','scale_fix','scale_fix_noaux'],'scale_fix':'Apply softplus before multiplying context stdev; dispersion never adds context mean, and divides by absolute affine scale. Private decode override, original source untouched.','noaux':'Same scale-fixed model with weight_l1 and weight_con zero; same architecture/data/update budget.','generation':'Native standard-normal latent-prior sampling retained. Training q versus generation prior difference is not treated as a bug. Native100samples; special10sample branch does not affect these scores.','stress_protocol':'After checkpoint frozen, add a fixed normalized level offset {-3,0,+3} to both history and target. This is a synthetic location-shift stress on real data, not a causal or full-pipeline unit-conversion experiment.','method_claim':False}
 p=out/'plan.json'
 plan['runtime_profile']={'threads':torch.get_num_threads(),'affinity':'0xffff0000' if 'affinity1' in tag else 'default','mkl_threading':os.environ.get('MKL_THREADING_LAYER','default')}
 if p.exists():assert json.loads(p.read_text())['training']==plan['training']
 else:dump(p,plan)
 raw=np.genfromtxt(ROOT/'data'/'ETTh1.csv',delimiter=',',skip_header=1,usecols=tuple(range(1,8)),dtype=np.float32)
 assert raw.shape[0]>=14400 and np.isfinite(raw).all()
 raw=torch.tensor(raw,device='cuda');mean=raw[:8640].mean(0);std=raw[:8640].std(0);data=(raw-mean)/(std+1e-9)
 val_orig=torch.arange(8640,8640+30*96,96,device='cuda');test_orig=torch.arange(11520,11520+30*96,96,device='cuda')
 np.savez(out/'split_scaler.npz',mean=mean.cpu().numpy(),std=std.cpu().numpy(),val_origins=val_orig.cpu().numpy(),test_origins=test_orig.cpu().numpy())
 results=[]
 for variant in plan['variants']:
  run=out/variant;run.mkdir(exist_ok=True)
  if (run/'result.json').exists():results.append(json.loads((run/'result.json').read_text()));continue
  model=build(seed=1)
  if variant!='released':model.model.decode=types.MethodType(fixed_decode,model.model)
  if variant=='scale_fix_noaux':model.weight_l1=0.;model.weight_con=0.
  opt=torch.optim.Adam(model.parameters(),lr=.001)
  sampler=torch.Generator(device='cuda');sampler.manual_seed(7101)
  best=float('inf');best_epoch=None;start=time.perf_counter()
  with (run/'train.jsonl').open('w',encoding='utf-8') as log:
   for epoch in range(1,51):
    total=0.
    # Isolate training RNG from validation MC calls.
    torch.manual_seed(8000+epoch)
    for _ in range(100):
     origins=torch.randint(192,8617,(48,),generator=sampler,device='cuda');batch=getbatch(data,origins)
     opt.zero_grad(set_to_none=True);loss=model.loss(batch);assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),.5);opt.step();total+=loss.item()
    metrics,_,_=score(model,data,val_orig,mean,std,seed=9101)
    if metrics['native_raw_CRPS_quantile19']<best:
     best=metrics['native_raw_CRPS_quantile19'];best_epoch=epoch;torch.save(model.state_dict(),run/'best.pt')
    row={'epoch':epoch,'train_loss':total/100,'validation':metrics,'selected_epoch_so_far':best_epoch,'elapsed_seconds':time.perf_counter()-start}
    log.write(json.dumps(row)+'\n');log.flush();dump(out/'status.json',{'state':'training','variant':variant,'epoch':epoch,'completed':len(results),'total':3})
    if epoch==1 or epoch%10==0:print(json.dumps({'variant':variant,**row}),flush=True)
  torch.save(model.state_dict(),run/'final.pt');model.load_state_dict(torch.load(run/'best.pt',map_location='cuda',weights_only=True))
  scores={}
  for offset in [-3.,0.,3.]:
   metric,sample,target=score(model,data,test_orig,mean,std,seed=9201,offset=offset);scores[str(offset)]=metric
   np.savez_compressed(run/f'test_offset{offset:+.0f}.npz',samples=sample,targets=target)
  row={'variant':variant,'seed':1,'epochs':50,'updates':5000,'parameters':sum(p.numel() for p in model.parameters()),'selected_epoch':best_epoch,'best_val_native_raw_CRPS':best,'seconds':time.perf_counter()-start,'checkpoint_sha256':hashlib.sha256((run/'best.pt').read_bytes()).hexdigest(),'test':scores}
  dump(run/'result.json',row);results.append(row);dump(out/'results.json',results);print(json.dumps(row),flush=True)
 dump(out/'status.json',{'state':'complete','completed':len(results),'total':3,'new_method_claim':False,'objective_achieved':False})
if __name__=='__main__':main()
