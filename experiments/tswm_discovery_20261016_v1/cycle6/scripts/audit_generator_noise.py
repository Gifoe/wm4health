"""Identify and isolate native generator noise reuse; preserve original dataset."""
import runtime_profile
import ast,copy,datetime,hashlib,json,os
import numpy as np
from scipy.stats import ortho_group
from scipy.linalg import block_diag
from native_child import ROOT,SOURCE,select
def innovations(path):
 with np.load(path/'data.npz') as data:y=data['yt']
 # The pinned source helper named leaky_ReLU returns D unchanged. Do not
 # substitute the standard activation merely because of the function's name.
 W=np.load(path/'W1.npy');base=y[:,:-1]@W
 current=y[:,1:];e=np.empty_like(current);e[:,:,0]=current[:,:,0]-base[:,:,0]
 for layer,size in enumerate([1,2,4]):
  if layer==0:continue
  start=sum([1,2,4][:layer]);parent_start=sum([1,2,4][:layer-1]);parent_sum=current[:,:,parent_start:start].sum(-1)
  for j in range(size):e[:,:,start+j]=current[:,:,start+j]-.2*base[:,:,start+j]-.8*j*parent_sum
 e=e.reshape(-1,7);cov=np.cov(e,rowvar=False);eigen=np.linalg.eigvalsh(cov);corr=np.corrcoef(e,rowvar=False);off=corr-np.eye(7)
 return {'count':len(e),'innovation_cov_eigenvalues':eigen.tolist(),'innovation_cov_rank_at_1e-8':int(np.sum(eigen>1e-8)),'maximum_absolute_offdiag_correlation':float(np.abs(off).max()),'noise0_vs_noise1_max_difference':float(np.abs(e[:,0]-e[:,1]).max()),'noise0_vs_noise3_max_difference':float(np.abs(e[:,0]-e[:,3]).max()),'noise2_vs_noise4_max_difference':float(np.abs(e[:,2]-e[:,4]).max()),'innovation_correlation':corr.tolist()}
def main():
 out=ROOT/'outputs/generator_noise_audit';out.mkdir(parents=True,exist_ok=True)
 native=ROOT/'data/native_A/A';original=innovations(native)
 (out/'native_only.json').write_text(json.dumps(original,indent=2),encoding='utf-8')
 print(json.dumps(original),flush=True)
 assert original['innovation_cov_rank_at_1e-8']==4 and original['noise0_vs_noise1_max_difference']<1e-12
 target=ROOT/'data/independent_noise_A';target.mkdir(parents=True,exist_ok=True);path=target/'A/data.npz'
 source=SOURCE/'tools/gen.py';names=['random_permutation_matrix','block_diagonal_permutation_matrix','leaky_ReLU_1d','leaky_ReLU','generateUniformMat','stationary_z_link']
 env={'np':np,'os':os,'ortho_group':ortho_group,'block_diag':block_diag,'root_dir':str(target)};select(source,names,env)
 tree=ast.parse(source.read_text(encoding='utf-8'));node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='stationary_z_link')
 class Repair(ast.NodeTransformer):
  count=0
  def visit_Subscript(self,n):
   self.generic_visit(n)
   if isinstance(n.value,ast.Name) and n.value.id=='y_t_noise' and isinstance(n.slice,ast.Tuple) and len(n.slice.elts)==2 and isinstance(n.slice.elts[1],ast.Name) and n.slice.elts[1].id=='i':
    self.count+=1;n.slice.elts[1]=ast.parse('sum(z_dim_list[:layer])+i',mode='eval').body
   return n
 repair=Repair();node=repair.visit(node);assert repair.count==3,repair.count
 # Three syntactic occurrences in the native function (two inactive inst=True
 # branch occurrences and the one active inst=False innovation use).
 exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(source),'exec'),env)
 if not path.exists():env['stationary_z_link'](4,2,False,'A',seed=772,z_dim_list=[1,2,4])
 repaired=innovations(target/'A');assert repaired['innovation_cov_rank_at_1e-8']==7
 result={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'native':original,'independent_noise_control':repaired,'control_definition':'Only use global hierarchical coordinate index in y_t_noise subscript; inst=False active change is one expression. Same seed772,100000 trajectories, graph/mixing functions unchanged. This is an independently labeled modified benchmark, not the original source data.','native_sha256':hashlib.sha256((native/'data.npz').read_bytes()).hexdigest(),'control_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'theorem_caution':'Independent innovations alone do not certify every sufficient-variability/smoothness/observability assumption; no theorem counterexample or new research gap is asserted.'}
 (out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k not in ['native','independent_noise_control']}));print('Native rank',original['innovation_cov_rank_at_1e-8'],'max correlation',original['maximum_absolute_offdiag_correlation']);print('Control rank',repaired['innovation_cov_rank_at_1e-8'],'max correlation',repaired['maximum_absolute_offdiag_correlation'])
if __name__=='__main__':main()
