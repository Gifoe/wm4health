"""One fixed-budget learned existing-model fit; only train/validation observations."""
from native61_v2 import *
import numpy as np,time,sys

def load_split(manifest,family,split):
 assert split in ['train','validation']
 r=manifest['families'][family][split];assert sha(r['file'])==r['sha256']
 with np.load(r['file']) as a:return a['X'].copy(),a['lengths'].copy()

def validation(m,x):
 m.eval();values=[]
 with torch.no_grad():
  for p in [32,64,96]:values.append(-(m.log_likelihood(x[:,:p+48])-m.log_likelihood(x[:,:p])).mean().item()/48)
 m.train();return float(np.mean(values))

def main(family,arm,seed):
 torch.set_num_threads(4);torch.use_deterministic_algorithms(True);q=read(OUT/'QUALIFICATION61_V2.json');data=read(OUT/'DATA61_V1.json');source=sources();device=q['chosen_device'];tag=f'{family}_{arm}_seed{seed}';run=OUT/'runs'/tag;run.mkdir(parents=True,exist_ok=False);ckdir=paths()['checkpoint_root']/tag;ckdir.mkdir(exist_ok=False)
 X,L=load_split(data,family,'train');V,_=load_split(data,family,'validation');x=torch.tensor(X,device=device);v=torch.tensor(V,device=device);m=model(arm,seed,device=device)
 with torch.no_grad():
  for name,p in m.named_parameters():
   if name.endswith('log_temperature'):p.zero_();p.requires_grad_(False)
  torch.manual_seed(seed+100)
  m._initialize_emission_from_observations(torch.cat([x[i,:int(n)] for i,n in enumerate(L)],0),cluster_variance=True)
 trainable=[p for p in m.parameters() if p.requires_grad];opt=torch.optim.Adam(trainable,lr=.01);rng=np.random.default_rng(seed+61000);curve=run/'curve.jsonl';chosen=None;start=time.perf_counter();lastloss=None
 def checkpoint(step,label):
  target=ckdir/f'{label}_step{step:05d}.pt'
  with target.open('xb') as f:torch.save({'model':m.state_dict(),'updates':step},f)
  return target
 initial=checkpoint(0,'initial')
 def choose(step):
  nonlocal chosen
  val=validation(m,v);assert np.isfinite(val),('Nonfinite validation',step,val)
  row={'utc':utc(),'step':step,'validation_conditional_nll_per_step':val,'last_training_nll_per_step':lastloss,'seconds':time.perf_counter()-start}
  if chosen is None or val<chosen['validation_conditional_nll_per_step']:
   file=checkpoint(step,'selected');chosen={**row,'checkpoint':str(file),'checkpoint_sha256':sha(file)}
  with curve.open('a',encoding='utf-8') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
  print(json.dumps({'tag':tag,**row}),flush=True)
 choose(0);completed=0;failed=False
 for step in range(1,1501):
  ids=rng.integers(0,len(L),size=16);batch=[x[int(i),:int(L[i])] for i in ids];opt.zero_grad();loss=-m.log_likelihood(batch).sum()/sum(int(L[i]) for i in ids);lastloss=float(loss.detach())
  if not np.isfinite(lastloss):failed=True;break
  loss.backward();norm=torch.nn.utils.clip_grad_norm_(trainable,5.)
  if not torch.isfinite(norm):failed=True;break
  opt.step();completed=step
  if step%100==0:choose(step)
 terminal=checkpoint(completed,'terminal');receipt={'utc':utc(),'status':'FAILED_NONFINITE_PRESERVED' if failed else 'COMPLETE_FIXED_BUDGET_NATIVE_DURATION_CONTROL','family':family,'arm':arm,'seed':seed,'tag':tag,'device':device,'updates_completed':completed,'parameters':sum(p.numel() for p in m.parameters()),'trainable_parameters':sum(p.numel() for p in trainable),'selected':chosen,'initial_checkpoint':str(initial),'initial_checkpoint_sha256':sha(initial),'terminal_checkpoint':str(terminal),'terminal_checkpoint_sha256':sha(terminal),'curve_sha256':sha(curve),'training_seconds':time.perf_counter()-start,'final_training_nll_per_step':lastloss if np.isfinite(lastloss) else None,'config':m.config.to_dict(),'adapter':'native61_v2.py with supported injected no-self transition for HSMM; native duration-one unrestricted HMM','adapter_sha256':sha(ROOT/'scripts/native61_v2.py'),'script_sha256':sha(__file__),'protocol_sha256':sha(ROOT/'PROTOCOL61_V1.md'),'source_receipt_sha256':sha(OUT/'SOURCES61_V1.json'),'data_manifest_sha256':sha(OUT/'DATA61_V1.json'),'read_roles':['train observations and lengths','validation observations only'],'audit_truth_read':False,'test_read':False,'test_score_selection':False,'overall_goal':'ACTIVE_UNACHIEVED','accepted_final_directions':0}
 dump(run/'result.json',receipt);print(json.dumps({'tag':tag,'status':receipt['status'],'selected_step':chosen['step'],'updates_completed':completed,'seconds':receipt['training_seconds']}),flush=True)
 if failed:sys.exit(3)

if __name__=='__main__':main(sys.argv[1],sys.argv[2],int(sys.argv[3]))
