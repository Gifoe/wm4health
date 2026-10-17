"""Replay published frozen-point gradients from typed JSON, without model training."""
from common54 import *
from parameter_card54_v1 import decode
from audit_em_switch54_v2 import function
import argparse,ctypes,time
fn=ctypes.windll.kernel32.SetProcessAffinityMask;fn.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
assert fn(ctypes.c_void_p(-1),ctypes.c_size_t(0xff))
def main():
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[0,1],default=0);a=p.parse_args();source_check();start=time.perf_counter()
 path=OUT/'training_audit_replay_v1'/f'rp_linear_small_seed{a.seed}_parameters.json';card=json.loads(path.read_text());assert sha(data_path(card['task']))==card['dataset_sha256'];params=decode(card['model_parameters'])
 with np.load(data_path(card['task'])) as ar:x=ar['train_x']
 rng=np.random.default_rng(5425);ids=np.concatenate([rng.permutation(200) for _ in range(4)]).reshape(25,32);A=jnp.asarray(params[0]['A']);rec=params[1];records=[]
 for em in [False,True]:
  _,full_fn=function(a.seed,200,em);_,g=full_fn(A,rec,jnp.asarray(x));full=float(np.sum(np.asarray(g)*np.asarray(A)));_,batch_fn=function(a.seed,32,em);gs=[]
  for i in ids:_,gb=batch_fn(A,rec,jnp.asarray(x[i]));gs.append(np.asarray(gb))
  batch=float(np.sum(np.mean(gs,axis=0)*np.asarray(A)));expected=card['expected_gradients'][str(em)];error=max(abs(full-expected['full_alpha_derivative']),abs(batch-expected['balanced_batch_alpha_derivative']));assert error<1e-7,error
  records.append({'em':em,'full_alpha_derivative':full,'balanced_batch_alpha_derivative':batch,'max_abs_error_vs_original':error})
 result={'utc':utc(),'status':'PASS_FROZEN_POINT_REPLAY_FROM_PUBLIC_JSON_PARAMETERS','seed':a.seed,'parameter_card_sha256':sha(path),'records':records,'runtime_seconds':time.perf_counter()-start,'optimizer_updates':0,'heldout_read':False,'latent_truth_read':False,'script_sha256':sha(__file__)}
 target=OUT/'training_audit_replay_v1'/f'replay_seed{a.seed}_verification.json';assert not target.exists();dump(target,result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
