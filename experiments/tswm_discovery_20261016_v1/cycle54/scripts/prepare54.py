"""Execute the pinned native linear generator; no training or evaluation."""
from common54 import *
from rp_ssm.datasets_utils import generate_linear_data
import importlib.metadata as md
def main():
 pins=source_check();records=[]
 for task in PLAN['tasks']:
  path=data_path(task['name']);assert not path.exists(),('Existing data retained; inspect manifest',path)
  data=generate_linear_data(1,task['latent_dim'],task['emission_dim'],PLAN['train_sequences'],PLAN['timesteps'],PLAN['generator_emission_cov_argument'],jr.PRNGKey(PLAN['generator_key']))
  arrays={'train_x':np.asarray(data.train_obs[0]),'train_z':np.asarray(data.train_states),'heldout_x':np.asarray(data.val_obs[0]),'heldout_z':np.asarray(data.val_states)}
  arrays.update({'true_'+k:np.asarray(v) for k,v in data.params.items()})
  assert arrays['train_x'].shape==(200,100,task['emission_dim'])
  assert arrays['heldout_x'].shape==(50,100,task['emission_dim'])
  assert np.max(abs(arrays['true_R']-.3*np.eye(task['emission_dim'])))<1e-14
  assert all(np.isfinite(x).all() for x in arrays.values())
  np.savez_compressed(path,**arrays)
  records.append({'task':task['name'],'file':str(path),'sha256':sha(path),'shapes':{k:list(v.shape) for k,v in arrays.items()},'actual_R_diagonal':np.diag(arrays['true_R']).tolist()})
 dump(OUT/'preflight54.json',{'utc':utc(),'status':'PASS_PINNED_NATIVE_SOURCE_AND_PAPER_LINEAR_GENERATOR','plan_sha256':sha(PLAN_PATH),'source_files_sha256':pins,'datasets':records,'python':sys.version,'device':[str(d) for d in jax.devices()],'packages':{k:md.version(k) for k in ['jax','jaxlib','numpy','flax','optax','dynamax']},'source_paper_reference':json.loads((PROJECT/'cycle2/outputs/literature/source_manifest.json').read_text(encoding='utf-8'))[0]})
 print('PREPARED',len(records),'native datasets',flush=True)
if __name__=='__main__':main()
