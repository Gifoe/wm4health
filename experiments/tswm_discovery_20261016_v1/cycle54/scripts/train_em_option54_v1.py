"""Matched unchanged-native RP fit with its existing em=True option."""
from common54 import *
import ctypes,argparse,pickle,time
from dataclasses import replace
from rp_ssm.datasets import TrainData
from rp_ssm.training import Trainer
fn=ctypes.windll.kernel32.SetProcessAffinityMask;fn.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
assert fn(ctypes.c_void_p(-1),ctypes.c_size_t(0xff000000))
def main():
 p=argparse.ArgumentParser();p.add_argument('task');p.add_argument('seed',type=int);a=p.parse_args();case(a.task);assert a.seed in [0,1];source_check()
 qual=json.loads((OUT/'em_switch_audit_v2/summary.json').read_text());assert qual['status']=='COMPLETE_EXISTING_NATIVE_EM_SWITCH_FALSIFICATION'
 root=OUT/'em_option_learning_v1';out=root/'runs'/f'rp_emtrue_{a.task}_seed{a.seed}';out.mkdir(parents=True,exist_ok=True);assert not (out/'result.json').exists()
 with np.load(data_path(a.task)) as ar:x=jnp.asarray(ar['train_x'])
 td=TrainData(obs=(x,));energy,cfg=rp_model(a.task,a.seed);cfg=replace(cfg,em=True);ref_energy,ref_cfg=rp_model(a.task,a.seed)
 _,init_key=jr.split(jr.PRNGKey(a.seed));initial,_,_=energy.init(init_key,td,cfg);reference,_,_=ref_energy.init(init_key,td,ref_cfg)
 init_arrays=jax.tree.map(lambda v:np.asarray(v),initial);ref_arrays=jax.tree.map(lambda v:np.asarray(v),reference)
 assert jax.tree.structure(init_arrays)==jax.tree.structure(ref_arrays)
 assert all(np.array_equal(v,w) and v.dtype==w.dtype for v,w in zip(jax.tree.leaves(init_arrays),jax.tree.leaves(ref_arrays)))
 original=CKPT/f'rp_{a.task}_seed{a.seed}_initial.pkl'
 if original.exists():
  orig=pickle.loads(original.read_bytes());assert all(np.array_equal(v,w) and np.asarray(v).dtype==np.asarray(w).dtype for v,w in zip(jax.tree.leaves(init_arrays),jax.tree.leaves(orig)))
 start=time.perf_counter();log=out/'curve.jsonl';assert not log.exists();run=out.name
 def logger(trainer,aux,indices):
  it=int(trainer.itr)+1
  if it==1 or it%100==0:
   value=float(trainer.loss_tot[-1]);A=np.asarray(trainer.params[0]['A']);assert np.isfinite(value) and np.isfinite(A).all()
   row={'iteration':it,'native_negative_auxiliary_free_energy':value,'elapsed_seconds':time.perf_counter()-start,'A_singular_values':np.linalg.svd(A,compute_uv=False).tolist()}
   with log.open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
   print(json.dumps(row),flush=True)
  return {}
 trainer=Trainer(energy,cfg,logger);trainer.fit(td,use_pbar=False);history=np.asarray(trainer.loss_tot);assert np.isfinite(history).all();params=jax.tree.map(lambda v:np.asarray(v),trainer.params);assert all(np.isfinite(v).all() for v in jax.tree.leaves(params))
 ckptdir=CKPT/'em_option_learning_v1';ckptdir.mkdir(exist_ok=True);cp=ckptdir/(run+'.pkl');ip=ckptdir/(run+'_initial.pkl');assert not cp.exists() and not ip.exists();cp.write_bytes(pickle.dumps(params));ip.write_bytes(pickle.dumps(init_arrays))
 artifact=ART/'em_option_learning_v1';artifact.mkdir(exist_ok=True);np.save(artifact/(run+'_objective.npy'),history)
 result={'utc':utc(),'status':'COMPLETE_FIXED_BUDGET_NATIVE_EM_TRUE','run':run,'task':a.task,'seed':a.seed,'method':'rp_emtrue','em':True,'iterations':cfg.num_iter,'batch_size':cfg.batch_size,'learning_rate':PLAN['learning_rate'],'checkpoint':str(cp),'checkpoint_sha256':sha(cp),'initial_checkpoint':str(ip),'initial_checkpoint_sha256':sha(ip),'original_false_init_arrays_exact_match':True,'matched_false_run':f'rp_{a.task}_seed{a.seed}','dataset_sha256':sha(data_path(a.task)),'original_plan_sha256':sha(PLAN_PATH),'supplemental_protocol_sha256':sha(ROOT/'EM_OPTION_LEARNING54_V1.md'),'native_source_revision':PLAN['source_revision'],'training_script_sha256':sha(__file__),'runtime_seconds':time.perf_counter()-start,'parameter_count':int(sum(v.size for v in jax.tree.leaves(params))),'checkpoint_selection':'terminal fixed-budget model','latent_truth_read':False,'heldout_read':False,'configuration_difference':'Only the existing native Config.em option changes gradient treatment; data/model/initialization/batch RNG/Adam/budget unchanged.'}
 dump(out/'result.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
