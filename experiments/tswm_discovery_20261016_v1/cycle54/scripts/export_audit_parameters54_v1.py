"""Publish only the two tiny frozen audit-point parameter trees, with exact roundtrip."""
from common54 import *
from parameter_card54_v1 import encode,decode
import pickle
def main():
 source_check();out=OUT/'training_audit_replay_v1';assert not out.exists();out.mkdir();records=[]
 for seed in [0,1]:
  run=f'rp_linear_small_seed{seed}';r=json.loads((OUT/'runs'/run/'result.json').read_text());assert sha(r['checkpoint'])==r['checkpoint_sha256'];params=pickle.loads(pathlib.Path(r['checkpoint']).read_bytes())
  tree=encode(params);restored=decode(json.loads(json.dumps(tree,allow_nan=False)))
  assert jax.tree.structure(params)==jax.tree.structure(restored)
  assert all(np.array_equal(v,w) and np.asarray(v).dtype==np.asarray(w).dtype for v,w in zip(jax.tree.leaves(params),jax.tree.leaves(restored)))
  gradient=json.loads((OUT/'em_switch_audit_v2'/f'{run}.json').read_text());card={'schema_version':1,'run':run,'task':'linear_small','seed':seed,'native_checkpoint_sha256':r['checkpoint_sha256'],'source_revision':PLAN['source_revision'],'dataset_sha256':r['dataset_sha256'],'model_parameters':tree,'all_parameter_arrays_and_dtypes_exact_after_json_roundtrip':True,'expected_gradients':{str(m['em']):{'full_alpha_derivative':m['full_alpha_derivative'],'balanced_batch_alpha_derivative':m['balanced_batch_alpha_derivative']} for m in gradient['modes']},'scope':'Two small public synthetic audit points only. Native runtime checkpoint files and all signal/score arrays remain on server storage. No model optimization or heldout access in this export.'}
  path=out/(run+'_parameters.json');dump(path,card);records.append({'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size,'native_checkpoint_sha256':r['checkpoint_sha256']})
 dump(out/'manifest.json',{'utc':utc(),'status':'TWO_AUDIT_PARAMETER_CARDS_EXACT_ARRAY_ROUNDTRIP','cards':records,'source_files_unchanged':source_check()});print(json.dumps({'cards':records,'status':'EXACT_ARRAY_ROUNDTRIP_PASS'}),flush=True)
if __name__=='__main__':main()
