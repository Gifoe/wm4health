"""Fresh objective comparison; complete held-out observed VAL selects all arms."""
from common47 import *
import argparse
torch.set_num_threads(1)
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--case',choices=CASES,required=True);parser.add_argument('--method',choices=METHODS,required=True);parser.add_argument('--seed',type=int,choices=SEEDS,required=True);a=parser.parse_args();qual=read(R/'outputs/qualification_v1/qualification.json');assert qual['status']=='QUALIFIED_FOR_FROZEN_FINITE_GRID_SCREEN';device=qual['selected_training_device'];folder=R/f'outputs/fits/{a.case}__{a.method}__seed{a.seed}';assert not folder.exists();folder.mkdir(parents=True)
 manifest=read(R/'data/MANIFEST.json')
 for name in [f'{a.case}/train.npz',f'{a.case}/val.npz',f'{a.case}/train_features.npz','feature_bank.npz']:assert sha(R/'data'/name)==manifest['files'][name]['sha256']
 train=np.load(R/'data'/a.case/'train.npz');val=np.load(R/'data'/a.case/'val.npz');stats=np.load(R/'data'/a.case/'train_features.npz');bank=np.load(R/'data/feature_bank.npz')['weights'];initial=torch.as_tensor(train['initial'],device=device);counts=torch.as_tensor(train['counts'],device=device,dtype=torch.float32);rho=torch.as_tensor(train['rho'],device=device);vi=torch.as_tensor(val['initial'],device=device);vc=torch.as_tensor(val['counts'],device=device,dtype=torch.float32);vr=torch.as_tensor(val['rho'],device=device);weights=torch.as_tensor(bank,device=device);means=torch.as_tensor(stats['mean'],device=device,dtype=torch.float32);variances=torch.as_tensor(stats['variance'],device=device,dtype=torch.float32);gen=np.random.default_rng(a.seed+100);batches=gen.integers(512,size=(6000,32));features=gen.integers(512,size=(6000,64));policies=gen.integers(2,size=6000);np.savez_compressed(folder/'schedule.npz',batches=batches,features=features,policies=policies)
 config={'case':a.case,'method':a.method,'seed':a.seed,'device':device,'source_sha256':sha(__file__),'common_sha256':sha(R/'scripts/common47.py'),'source_component_sha256':sha(component_path),'protocol_sha256':sha(R/'PROTOCOL.md'),'data_manifest_sha256':sha(R/'data/MANIFEST.json'),'schedule_sha256':sha(folder/'schedule.npz'),'created_utc':now(),'original_paper_headline_reproduction':False};seed(a.seed);model=World(a.method=='mean_nll').to(device);config['parameters']=sum(p.numel() for p in model.parameters());config['initial_weight_checksum']=checksum(model);dump(folder/'config.json',config);opt=torch.optim.Adam(model.parameters(),lr=1e-3);best=float('inf');best_step=0;stale=0;validation=[];reason=[];losses=[];start=time.perf_counter()
 for i in range(6000):
  step=i+1;lr=1e-3*(.1+.9*.5*(1+math.cos(math.pi*i/6000)))
  for group in opt.param_groups:group['lr']=lr
  index=batches[i];findex=features[i];pi=int(policies[i]);model.train();opt.zero_grad(set_to_none=True);loss,parts=objective(model,a.method,initial[index],counts[index],rho[index],weights[findex],means[pi,findex],variances[pi,findex],RHOS[pi]);assert torch.isfinite(loss),step;loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.);assert torch.isfinite(norm),step;opt.step();losses.append(float(loss.detach().cpu()))
  if step%50==0:dump(folder/'status.json',{'status':'training','pid':os.getpid(),'step':step,'best_step':best_step,'best_val_nll':best if math.isfinite(best) else None,'elapsed_seconds':time.perf_counter()-start,'utc':now()})
  if step%200:continue
  model.eval();values=[]
  with torch.no_grad():
   P,rate=model.components()
   for j in range(0,len(vi),32):values.extend((-model.joint(vi[j:j+32],vc[j:j+32],vr[j:j+32],P,rate)/(T+1)).cpu().tolist())
  value=float(np.mean(values));previous=best;relative=(previous-value)/max(abs(previous),1e-8) if math.isfinite(previous) else None
  if value<best:best=value;best_step=step;torch.save({'model':model.state_dict(),'step':step,'val_nll':value,'config':config},folder/'selected.pt')
  stale=0 if relative is None or relative>=.001 else stale+1;row={'step':step,'mean_train_loss_last_200':float(np.mean(losses)),'val_nll_per65':value,'best_val_nll':best,'best_step':best_step,'relative_best_improvement':relative,'stale_checks':stale,'elapsed_seconds':time.perf_counter()-start,'lr':lr};validation.append(row);dump(folder/'validation.json',validation);losses=[];print(json.dumps({'tag':folder.name,**row}),flush=True)
  if step>=1200 and stale>=3:reason.append('three_checks_below_0.1pct_after_minimum');break
  if step==3000:
   extend=best_step==3000 or any((r['relative_best_improvement'] or 0)>=.001 for r in validation[-2:]);reason.append('soft_cap_extended' if extend else 'soft_cap_plateau')
   if not extend:break
 torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'step':step,'config':config,'rng_cpu':torch.random.get_rng_state()},folder/'last.pt');result={'status':'FIT_COMPLETE_EVAL_UNOPENED','tag':folder.name,'case':a.case,'method':a.method,'seed':a.seed,'updates':step,'selected_step':best_step,'selected_val_nll':best,'selected_at_final_update':best_step==step,'hard_cap_hit':step==6000,'stop_reasons':reason,'elapsed_seconds':time.perf_counter()-start,'selected_sha256':sha(folder/'selected.pt'),'last_sha256':sha(folder/'last.pt'),'schedule_sha256':config['schedule_sha256'],'initial_weight_checksum':config['initial_weight_checksum'],'last_weight_checksum':checksum(model),'utc':now()};dump(folder/'fit_summary.json',result);dump(folder/'status.json',{'status':'completed','step':step,'utc':now()})
if __name__=='__main__':main()
