"""Complete native A config/model/loss under disclosed manual Torch orchestration."""
import runtime_profile
import datetime,hashlib,json,os,pathlib,time
from types import SimpleNamespace
import numpy as np,torch
import scipy.stats as stats
from scipy.optimize import linear_sum_assignment
from native_child import ROOT,build
def dump(p,o):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(o,indent=2),encoding='utf-8')
def evaluate(model,x,y):
 with torch.no_grad():
  target,recon,mus,logvar,z,emb=model.forward({'xt':x,'yt':y})
 m=mus.detach().cpu().reshape(-1,7).numpy();true=y[:,2:-2].detach().cpu().reshape(-1,7).numpy()
 corr=stats.spearmanr(true,m).statistic[:7,7:];a,b=linear_sum_assignment(-np.abs(corr));mcc=float(np.abs(corr[a,b]).mean())
 return {'mcc_spearman':mcc,'reconstruction_MSE':(recon-target).square().mean().item(),'max_abs_encoded_mean':mus.abs().max().item()},m,true
def main():
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
 tag=os.environ.get('TSWM_CHILD_RUN','child_native_A');assert '/' not in tag and '\\' not in tag and tag not in ['.','..']
 out=ROOT/'outputs'/tag;out.mkdir(parents=True,exist_ok=True)
 if (out/'plan.json').exists():raise RuntimeError('Existing run: inspect process/status rather than duplicate')
 plan={'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'CLOSE source/config execution: complete native A100000 trajectories/10epochs, unchanged full neural class/loss/optimizer bodies; manual Torch orchestration/nn.Module container adapter, offline; not exact Lightning launcher reproduction','source_commit':'28cc8a8a702018c9c54bd02e26cbd8ccf756ab99','data_seed':772,'training_seed':770,'epochs':10,'batch_size':64,'train_count':98976,'validation_count':1024,'optimizer':'Native AdamW3e-4,betas(.9,.999),weight_decay1e-4','checkpoint':'Maximum validation Spearman MCC, same synthetic protocol uses latent labels for validation only; no independent test claim','validation':'Twice per epoch; validation sample1024, native posterior means, same permutation-invariant Spearman MCC via scipy assignment','external_logging':'No external logging or launcher login executed','native_data_assumption_warning':'Native A shares innovation indices across hierarchical layers. Its identifiability-assumption fidelity must be audited separately.'}
 dump(out/'plan.json',plan);torch.manual_seed(770);np.random.seed(770)
 with np.load(ROOT/'data/native_A/A/data.npz') as data:x=torch.tensor(data['xt'],device='cuda',dtype=torch.float32);y=torch.tensor(data['yt'],device='cuda',dtype=torch.float32)
 split=torch.randperm(100000);train=split[:98976].to('cuda');valid=split[98976:].to('cuda');np.savez(out/'split_indices.npz',train=train.cpu().numpy(),validation=valid.cpu().numpy())
 model=build();optimizer=model.configure_optimizers()[0][0];best=-1.;best_epoch=0;start=time.perf_counter();step=0
 with (out/'train.jsonl').open('w',encoding='utf-8') as log:
  for epoch in range(1,11):
   order=train[torch.randperm(len(train),device='cuda')];nbatches=(len(order)+63)//64;total=0.;t0=time.perf_counter()
   for batch_index in range(nbatches):
    ids=order[batch_index*64:(batch_index+1)*64];model.train();optimizer.zero_grad(set_to_none=True)
    target,recon,mus,logvars,z,emb=model.forward({'xt':x[ids],'yt':y[ids]})
    _,rec,normal,future=model.loss_function(target,recon,mus,logvars,z,emb);loss=.1*rec+model.beta*normal+model.gamma*future
    assert torch.isfinite(loss),(epoch,batch_index,float(loss));loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters());optimizer.step();total+=loss.item();step+=1
    if batch_index+1 in {nbatches//2,nbatches}:
     model.eval();metrics,_,_=evaluate(model,x[valid],y[valid])
     if metrics['mcc_spearman']>best:best=metrics['mcc_spearman'];best_epoch=epoch;torch.save(model.state_dict(),out/'best.pt')
     row={'epoch':epoch,'batch':batch_index+1,'global_step':step,'train_loss_mean_so_far':total/(batch_index+1),'validation':metrics,'best_val_mcc':best,'elapsed_seconds':time.perf_counter()-start}
     log.write(json.dumps(row)+'\n');log.flush();dump(out/'status.json',{'state':'training','epoch':epoch,'step':step,'best_val_mcc':best,'objective_achieved':False});print(json.dumps(row),flush=True)
   torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},out/'resume.pt')
 torch.save(model.state_dict(),out/'final.pt');model.load_state_dict(torch.load(out/'best.pt',map_location='cuda',weights_only=True));model.eval();metric,m,true=evaluate(model,x[valid],y[valid]);np.savez_compressed(out/'validation_encoding.npz',mus=m,true=true)
 result={'state':'complete','epochs':10,'updates':step,'parameters':sum(p.numel() for p in model.parameters()),'best_validation_mcc':best,'best_epoch':best_epoch,'final_recheck':metric,'seconds':time.perf_counter()-start,'new_gap_accepted':False,'objective_achieved':False,'scope':plan['scope']};dump(out/'result.json',result);dump(out/'status.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
