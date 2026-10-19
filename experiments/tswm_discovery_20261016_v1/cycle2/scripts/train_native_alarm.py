"""Native RP-GSSM learning on a new controlled hidden-state diagnostic.

Upstream training, loss, recognition MLP and posterior code are unmodified.
The true AR prior is fixed by a zero learning rate. Latent states never enter
the learning objective or checkpoint selection. Numerical inference follows
in a separate script after all model checkpoints have been selected.
"""
import os
os.environ['JAX_PLATFORM_NAME']='cpu';os.environ['JAX_ENABLE_X64']='true'
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
if os.name=='nt':
 import ctypes
 fn=ctypes.windll.kernel32.SetProcessAffinityMask;fn.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
 assert fn(ctypes.c_void_p(-1),ctypes.c_size_t(0xffff0000))
import argparse,copy,hashlib,json,pathlib,pickle,sys,time
import numpy as onp
from scipy.special import expit
import jax
import jax.numpy as np
import jax.random as jr
ROOT=pathlib.Path(__file__).resolve().parents[1];SRC=ROOT.parent/'upstream/rp-gssm'
sys.path.insert(0,str(SRC))
from rp_ssm.config import Config
from rp_ssm.datasets import TrainData
from rp_ssm.distmaps import MVNDiag
from rp_ssm.dists import LGStationaryParam
from rp_ssm.recognition import MLP,RPMRecognition
from rp_ssm.rpm import RPSSM,ConstrainedIVFreeEnergy
from rp_ssm.training import Trainer

def make_data(plan,kind,split):
 rng=onp.random.default_rng(plan['data_seeds'][split]);N=plan[{'train':'train_sequences','val':'validation_sequences','test':'test_sequences'}[split]];T=plan['timesteps'];rho=plan['rho']
 z=onp.empty((N,T));z[:,0]=rng.normal(size=N)
 for t in range(1,T):z[:,t]=rho*z[:,t-1]+onp.sqrt(1-rho*rho)*rng.normal(size=N)
 mask=rng.random((N,T))<plan['anchor_probability'];sensor=z+plan['anchor_noise_sd']*rng.normal(size=(N,T))
 probability=expit(plan['alarm_intercept']+plan['alarm_quadratic_coefficient']*z*z) if kind=='state_dependent_alarm' else onp.full_like(z,plan['independent_alarm_probability'])
 alarm=rng.random((N,T))<probability
 x=onp.stack([sensor*mask,mask,alarm],axis=-1).astype('float64')
 return {'obs':x,'z':z,'mask':mask,'alarm':alarm}

def construct(plan,seed):
 prior=LGStationaryParam(start_from_invariant=True,stay_at_invariant=True,opt_params=['A'],A=np.array([[plan['rho']]]))
 rec=RPMRecognition(network=MLP(tuple(plan['recognition_hidden_widths'])),dist_map=MVNDiag(1))
 energy=ConstrainedIVFreeEnergy(RPSSM(prior,[rec]))
 conf=Config(batch_size=plan['batch_size'],num_iter=plan['training_iterations'],seed=seed,prior_lr=0.,rec_lr=(plan['recognition_lr'],),stabilize_A=None,em=plan['em'])
 return energy,conf

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--plan',default='native_learning_plan.json');args=parser.parse_args()
 pp=ROOT/'outputs'/args.plan;plan=json.loads(pp.read_text(encoding='utf-8-sig'));out=ROOT/'outputs'/plan.get('output_subdirectory','native_alarm');out.mkdir(exist_ok=True,parents=True)
 logpath=out/'training.jsonl';allrows=[];start=time.perf_counter()
 for kind in plan['datasets']:
  arrays={split:make_data(plan,kind,split) for split in ['train','val','test']}
  dataset_path=out/(kind+'_data.npz')
  if not dataset_path.exists():onp.savez_compressed(dataset_path,**{split+'_'+k:v for split,items in arrays.items() for k,v in items.items()})
  for seed in plan['training_seeds']:
   stem=kind+'_seed'+str(seed);result_file=out/(stem+'.json')
   if result_file.exists():
    print('completed result retained',stem,flush=True);continue
   energy,conf=construct(plan,seed);data=TrainData(obs=(np.array(arrays['train']['obs']),));val=TrainData(obs=(np.array(arrays['val']['obs']),))
   best={'loss':onp.inf,'iteration':None,'params':None};tstart=time.perf_counter();rows=[]
   val_loss=jax.jit(lambda params,batch:energy.loss(params,batch,1.,False,jr.PRNGKey(0))[0])
   def evaluate(trainer,iteration):
    losses=[float(val_loss(trainer.params,val[i:i+conf.batch_size])) for i in range(0,len(arrays['val']['obs']),conf.batch_size)]
    vl=float(onp.mean(losses));assert onp.isfinite(vl)
    if vl<best['loss']:
     best.update(loss=vl,iteration=iteration,params=jax.tree.map(lambda a:onp.asarray(a).copy(),trainer.params))
    return vl
   def logger(trainer,aux,indices):
    iteration=int(trainer.itr+1)
    if iteration==1 or iteration%100==0:
     train_loss=float(trainer.loss_tot[-1]);assert onp.isfinite(train_loss)
     row={'dataset':kind,'seed':seed,'iteration':iteration,'native_train_loss':train_loss,'elapsed_seconds':time.perf_counter()-tstart}
     if iteration==1 or iteration%200==0:row['native_validation_loss']=evaluate(trainer,iteration)
     rows.append(row)
     with logpath.open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
     print(json.dumps(row),flush=True)
    return {}
   trainer=Trainer(energy,conf,logger=logger);trainer.fit(data,use_pbar=False)
   assert best['params'] is not None
   ckpt=out/(stem+'.pkl');ckpt.write_bytes(pickle.dumps(best['params']))
   result={'dataset':kind,'seed':seed,'selected_iteration':best['iteration'],'selected_native_validation_loss':best['loss'],
    'iterations_completed':conf.num_iter,'checkpoint_sha256':hashlib.sha256(ckpt.read_bytes()).hexdigest(),'data_sha256':hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
    'plan_sha256':hashlib.sha256(pp.read_bytes()).hexdigest(),'code_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
    'native_source_revision':plan['native_revision'],'runtime_seconds':time.perf_counter()-tstart,
    'dataset_summary':{split:{'anchor_fraction':float(items['mask'].mean()),'alarm_fraction':float(items['alarm'].mean()),'state_variance':float(items['z'].var())} for split,items in arrays.items()},'training_rows':rows}
   result_file.write_text(json.dumps(result,indent=2),encoding='utf-8');allrows.append(result)
 print('all native training completed',time.perf_counter()-start,flush=True)
if __name__=='__main__':main()
