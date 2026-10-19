"""Eight fresh ordinary controls. Selection uses seen-cell VAL only."""
from common46 import *
import argparse,datetime,os
torch.set_num_threads(1)

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
 p=argparse.ArgumentParser();p.add_argument('--obs',type=int,required=True);p.add_argument('--dyn',type=int,required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args();assert a.obs in (0,1) and a.dyn in (0,1) and a.seed in SEEDS
 qual=read(R/'outputs/qualification_v1/qualification.json');assert qual['status'].startswith('QUALIFIED');device=qual['selected_training_device'];tag=f'o{a.obs}d{a.dyn}_seed{a.seed}';out=R/'outputs/fits'/tag;assert not out.exists();out.mkdir(parents=True)
 frozen=read(R/'data/MANIFEST.json')
 for name in ['train.npz','val.npz','anchors.npz','normalizers.json']:assert sha(R/'data'/name)==frozen['files'][name]
 train=dict(np.load(R/'data/train.npz'));val=dict(np.load(R/'data/val.npz'));anchors=np.load(R/'data/anchors.npz')['X'];keys=sorted(train);schedule_rng=np.random.default_rng(a.seed+100);schedule=[]
 for i in range(8000):
  anchor=(i+1)%5==0;k=int(schedule_rng.integers(len(keys)));s=int(schedule_rng.integers(4));indices=schedule_rng.integers(64,size=16);schedule.append((anchor,k,s,indices))
 np.savez_compressed(out/'schedule.npz',anchor=np.array([r[0] for r in schedule]),cell=np.array([r[1] for r in schedule]),anchor_system=np.array([r[2] for r in schedule]),index=np.stack([r[3] for r in schedule]))
 config={'tag':tag,'obs':a.obs,'dyn':a.dyn,'seed':a.seed,'device':device,'common_sha256':sha(R/'scripts/common46.py'),'adapter_sha256':sha(R/'scripts/joint_adapter46.py'),'train_source_sha256':sha(__file__),'protocol_sha256':sha(R/'PROTOCOL.md'),'data_manifest_sha256':sha(R/'data/MANIFEST.json'),'schedule_sha256':sha(out/'schedule.npz'),'created_utc':now(),'interpretation':'ordinary source-core adaptation, not original-paper reproduction or a new method'};dump(out/'config.json',config)
 seed(a.seed);model=World(a.obs,a.dyn).to(device);optimizer=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-5);best=float('inf');best_step=0;stale=0;validation=[];start=time.perf_counter();updates=0;reasons=[];val_draw_seed=a.seed+20000;loss_buffer=[];last_ext=False
 initial_checksum=checksum(model);config.update(parameters=sum(p.numel() for p in model.parameters()),initial_weight_checksum=initial_checksum);dump(out/'config.json',config)
 for i,(anchor,k,s,idx) in enumerate(schedule):
  step=i+1;lr=1e-3*(.1+.9*.5*(1+math.cos(math.pi*i/8000)))
  for group in optimizer.param_groups:group['lr']=lr
  key=keys[k];view=None if anchor else int(key[1]);system=s if anchor else int(key[-1]);batch=anchors[idx] if anchor else train[key][idx];y=torch.as_tensor(batch,device=device);model.train();optimizer.zero_grad(set_to_none=True);loss=model.loss(y,view,system)
  if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite objective at {step}')
  loss.backward();gradnorm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.);assert torch.isfinite(gradnorm);optimizer.step();updates=step;loss_buffer.append(float(loss.detach().cpu()))
  if step%50==0:dump(out/'status.json',{'status':'training','pid':os.getpid(),'step':step,'best_step':best_step,'best_val_energy':best if math.isfinite(best) else None,'elapsed_seconds':time.perf_counter()-start,'utc':now()})
  if step%250:continue
  scores=[]
  with torch.no_grad():
   for j,key in enumerate(keys):
    draw=model.sample(val[key][:8,:HIST],int(key[1]),int(key[-1]),32,val_draw_seed+j);scores.extend(energy(draw,val[key][:8,HIST:]).cpu().tolist())
  score=float(np.mean(scores));previous=best;relative=(previous-score)/max(abs(previous),1e-8) if math.isfinite(previous) else None;improved=score<best
  if improved:best=score;best_step=step;torch.save({'model':model.state_dict(),'step':step,'val_energy':score,'seed':a.seed,'config':config},out/'selected.pt')
  stale=0 if relative is None or relative>=.002 else stale+1
  row={'step':step,'mean_train_loss_last_250':float(np.mean(loss_buffer)),'val_energy':score,'best_val_energy':best,'best_step':best_step,'relative_best_improvement':relative,'stale_checks':stale,'elapsed_seconds':time.perf_counter()-start,'lr':lr};loss_buffer=[];validation.append(row);dump(out/'validation.json',validation);print(json.dumps({'tag':tag,**row}),flush=True)
  if step>=1000 and stale>=3:reasons.append('three_checks_below_0.2pct_best_improvement_after_minimum');break
  if step==4000:
   last_ext=best_step==4000 or any((r['relative_best_improvement'] or 0)>=.002 for r in validation[-2:]);reasons.append('soft_cap_extended' if last_ext else 'soft_cap_plateau')
   if not last_ext:break
 torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':updates,'seed':a.seed,'rng_cpu':torch.random.get_rng_state(),'config':config},out/'last.pt');dump(out/'fit_summary.json',{'status':'FIT_COMPLETE_EVAL_UNOPENED','tag':tag,'utc':now(),'updates':updates,'selected_step':best_step,'selected_val_energy':best,'selected_at_final_step':best_step==updates,'hard_cap_hit':updates==8000,'stop_reasons':reasons,'elapsed_seconds':time.perf_counter()-start,'initial_checksum':initial_checksum,'last_checksum':checksum(model),'selected_sha256':sha(out/'selected.pt'),'last_sha256':sha(out/'last.pt'),'parameters':config['parameters'],'source_sha256':config['train_source_sha256'],'schedule_sha256':config['schedule_sha256']});dump(out/'status.json',{'status':'completed','step':updates,'utc':now()})
if __name__=='__main__':main()
