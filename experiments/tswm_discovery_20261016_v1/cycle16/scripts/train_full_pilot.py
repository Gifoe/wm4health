"""Native full MTS3 calendar-only forecasting pilot, with complete checkpoints/logs."""
import runtime_profile
import datetime,hashlib,json,pathlib,sys,time
import numpy as np
import torch
from config_tools import load
from data_protocol import Data,L,H,T,metrics,sha
ROOT=pathlib.Path(__file__).resolve().parents[1];SOURCE=ROOT.parent/'cycle15/upstream/MTS3'
sys.path.insert(0,str(SOURCE))
from agent.worldModels.MTS3 import MTS3
SEED=2021
def atomic(path,value):
 temporary=path.with_name(path.name+'.tmp');temporary.write_text(json.dumps(value,indent=2),encoding='utf-8');temporary.replace(path)
def main():
 out=ROOT/'outputs/train/NP_MTS3_seed2021';assert not out.exists();out.mkdir(parents=True)
 data=Data();manifest=json.loads((ROOT.parent/'cycle15/outputs/source/manifest.json').read_text())
 for row in manifest['files']:assert sha(SOURCE/row['path'])==row['sha256']
 preflight=json.loads((ROOT/'outputs/preflight.json').read_text());assert preflight['native_forward_backward_finite'] and max(preflight['future_placeholder_errors'])==0
 torch.set_num_threads(1);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 rng=np.random.default_rng(SEED);native_config=load(SOURCE/'experiments/mobileRobot/conf/model/default_mts3.yaml')
 model=MTS3(input_shape=(1,),action_dim=6,config=native_config).cuda();optimizer=torch.optim.Adam(model.parameters(),lr=.003)
 config={'model':'native_MTS3','dataset':'NP','seed':SEED,'history':L,'horizon':H,'source_steps':T,'batch_size':16,'lr':.003,
         'maximum_updates':1500,'validation_every':100,'patience_validation_checks':8,'gradient_clip_norm':5.,'selection_metric':'validation_NLL',
         'source_revision':manifest['revision'],'native_model_config':dict(native_config),'parameters':sum(p.numel() for p in model.parameters()),
         'data_registry_sha256':sha(ROOT/'outputs/data/registry.json'),'source_csv_sha256':data.registry['source_sha256'],
         'plan_sha256':sha(ROOT/'FULL_MODEL_PLAN.md'),'script_sha256':sha(pathlib.Path(__file__)),
         'adapter_sha256':sha(ROOT/'scripts/data_protocol.py'),'config_adapter_sha256':sha(ROOT/'scripts/config_tools.py'),
         'information':'Historical target and deterministic calendar only; no future raw forecast columns; no test labels',
         'scope':'New cross-benchmark native-framework transfer pilot; not original paper reproduction',
         'environment':{'python':sys.version,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'tf32':False}}
 atomic(out/'config.json',config);start=time.perf_counter();best=float('inf');best_step=0;bad=0;valrows=[]
 def batch(origins,partition):
  return [torch.as_tensor(a,device='cuda') for a in data.batch(origins,partition)]
 def evaluate(step):
  model.eval();means=[];variances=[];targets=[]
  with torch.inference_mode():
   for i in range(0,len(data.val),16):
    obs,cond,mask,target=batch(data.val[i:i+16],'val');m,v=model(obs,cond,mask)[:2]
    means.append(m[:,L-1:T].cpu().numpy());variances.append(v[:,L-1:T].cpu().numpy());targets.append(target.cpu().numpy())
  mean=np.concatenate(means);var=np.concatenate(variances);target=np.concatenate(targets)
  row={'step':step,**metrics(mean,var,target),'elapsed_seconds':time.perf_counter()-start,'validation_only':True,'no_test_evaluation':True}
  np.savez_compressed(out/f'validation_step{step:06d}.npz',origins=data.val,mean=mean,variance=var,target=target)
  with (out/'validation.jsonl').open('a',encoding='utf-8') as log:log.write(json.dumps(row)+'\n')
  valrows.append(row);print(json.dumps({'validation':row}),flush=True);model.train();return row
 def checkpoint(step):
  path=out/f'checkpoint_step{step:06d}.pt';assert not path.exists()
  payload={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step,'seed':SEED,
           'torch_rng_cpu':torch.get_rng_state(),'torch_rng_cuda':torch.cuda.get_rng_state_all(),
           'numpy_rng_json':json.dumps(rng.bit_generator.state),'config_sha256':sha(out/'config.json')}
  temporary=path.with_name(path.name+'.tmp');torch.save(payload,temporary);temporary.replace(path)
  atomic(out/'last_checkpoint.json',{'step':step,'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path)})
  return path
 path=checkpoint(0);row=evaluate(0);best=row['NLL'];atomic(out/'best_checkpoint.json',{'step':0,'NLL':best,'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path)})
 torch.cuda.reset_peak_memory_stats();model.train();step=0
 with (out/'training.jsonl').open('x',encoding='utf-8') as log:
  for step in range(1,1501):
   step_start=time.perf_counter();origins=rng.choice(data.train,size=16,replace=True)
   obs,cond,mask,target=batch(origins,'train');optimizer.zero_grad(set_to_none=True)
   mean,var=model(obs,cond,mask)[:2];mean=mean[:,L-1:T];var=var[:,L-1:T]
   assert torch.isfinite(mean).all() and torch.isfinite(var).all() and (var>0).all()
   loss=.5*(torch.log(2*torch.pi*var)+(target-mean)**2/var);loss=loss.mean();assert torch.isfinite(loss)
   loss.backward();gradnorm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.);assert torch.isfinite(gradnorm)
   optimizer.step()
   entry={'step':step,'NLL':float(loss.detach()),'gradient_norm_before_clip':float(gradnorm),
          'seconds':time.perf_counter()-step_start,'origins':origins.tolist(),'elapsed_seconds':time.perf_counter()-start}
   log.write(json.dumps(entry)+'\n');log.flush()
   if step==1 or step%20==0:print(json.dumps({'train_step':step,'NLL':entry['NLL'],'gradient_norm_before_clip':entry['gradient_norm_before_clip'],'elapsed_seconds':entry['elapsed_seconds']}),flush=True)
   if step%100==0:
    path=checkpoint(step);row=evaluate(step)
    if row['NLL']<best:
     best=row['NLL'];best_step=step;bad=0
     atomic(out/'best_checkpoint.json',{'step':step,'NLL':best,'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path)})
    else:bad+=1
    atomic(out/'progress.json',{'status':'training','step':step,'best_step':best_step,'best_validation_NLL':best,'nonimproving_checks':bad,
                              'elapsed_seconds':time.perf_counter()-start,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'objective_achieved':False})
    if bad>=8:break
 result={'training_complete':True,'last_step':step,'best_step':best_step,'best_validation_NLL':best,'validation':valrows,
         'elapsed_seconds':time.perf_counter()-start,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
         'stopping_reason':'patience' if bad>=8 else 'pilot_update_budget','training_adequacy_not_proven_by_budget':True,
         'no_test_evaluation':True,'single_seed_pilot':True,'new_method_claim':False,'objective_achieved':False,
         'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 atomic(out/'results.json',result);atomic(out/'progress.json',{'status':'completed',**result});print(json.dumps(result),flush=True)
if __name__=='__main__':main()
