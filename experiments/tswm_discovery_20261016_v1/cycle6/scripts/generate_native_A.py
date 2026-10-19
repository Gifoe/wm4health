"""Run exactly the native A generator function; do not generate the whole suite."""
import runtime_profile
import ast,datetime,hashlib,json,os,pathlib
import numpy as np
from scipy.stats import ortho_group
from scipy.linalg import block_diag
from native_child import ROOT,select,SOURCE
def main():
 out=ROOT/'data/native_A';out.mkdir(parents=True,exist_ok=True)
 path=out/'A/data.npz'
 (ROOT/'outputs').mkdir(parents=True,exist_ok=True)
 env={'np':np,'os':os,'ortho_group':ortho_group,'block_diag':block_diag,'root_dir':str(out)}
 source=SOURCE/'tools/gen.py'
 names=['random_permutation_matrix','block_diagonal_permutation_matrix','leaky_ReLU_1d','leaky_ReLU','generateUniformMat','stationary_z_link']
 select(source,names,env)
 if not path.exists():env['stationary_z_link'](4,2,False,'A',seed=772,z_dim_list=[1,2,4])
 else:print('Preserving generated native A; validating and recording provenance')
 with np.load(path) as data:
  assert data['xt'].shape==(100000,7,4) and data['yt'].shape==(100000,7,7)
  assert np.isfinite(data['xt']).all() and np.isfinite(data['yt']).all()
 record={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_commit':'28cc8a8a702018c9c54bd02e26cbd8ccf756ab99','source_function':'stationary_z_link(4,2,False,A,seed=772,z_dim_list=[1,2,4])','function_body_unchanged':True,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'data_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size,'scope':'Native benchmark data, including native noise-indexing behavior; not certified independent innovations'}
 (ROOT/'outputs/native_A_provenance.json').write_text(json.dumps(record,indent=2),encoding='utf-8');print(json.dumps(record),flush=True)
if __name__=='__main__':main()
