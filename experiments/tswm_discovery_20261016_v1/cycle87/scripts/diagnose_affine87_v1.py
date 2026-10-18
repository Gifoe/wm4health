from common87_v1 import *
from ordinary87_v1 import identify,FAMILIES
from scipy.linalg import expm
import numpy as np
def main():
 result=read(O/'EVALUATION_RESULTS87_V2.json');ordinary=read(O/'ORDINARY_SELECTION87_V1.json');rows=[]
 for row in result['rows']:
  if row['model']!='ordinary_weak_polynomial':continue
  with np.load(RAW/'evaluation_v2'/f"{row['family']}_{row['cohort']}_v1.npz") as z:d={k:z[k] for k in z.files}
  sl=slice(FAMILIES.index(row['family'])*128,(FAMILIES.index(row['family'])+1)*128)
  with np.load(row['array_path']) as z:p=z['prediction'][sl]
  coef=identify(d['context'],1,ordinary['selected']['ridge']);aug=np.zeros((128,3,3));aug[:,:2,:2]=np.stack([coef[:,2,:],coef[:,1,:]],axis=-1);aug[:,:2,2]=coef[:,0,:];transition=np.stack([expm(a*.05) for a in aug]);state=np.concatenate([d['query_initial'],np.ones((128,1))],axis=1);parts=[state[:,:2].copy()]
  for _ in range(200):state=np.einsum('bij,bj->bi',transition,state);parts.append(state[:,:2].copy())
  exact=np.stack(parts,axis=1);ok=np.isfinite(p).all(axis=(1,2));errors=np.max(abs(exact-p)/np.maximum(abs(exact),1),axis=(1,2));k=int(np.nanargmax(np.where(ok,errors,np.nan)));entry=dict(family=row['family'],cohort=row['cohort'],finite=int(ok.sum()),worst_case=k,max_scaled=float(errors[k]),max_abs=float(abs(exact[k]-p[k]).max()),matrix=aug[k].tolist(),eigenvalues_real=np.linalg.eigvals(aug[k,:2,:2]).real.tolist(),eigenvalues_imag=np.linalg.eigvals(aug[k,:2,:2]).imag.tolist(),exact_final=exact[k,-1].tolist(),stored_final=p[k,-1].tolist(),source_array_sha256=sha(row['array_path']));rows.append(entry);log(dict(stage='affine_integrator_diagnostic',**entry))
 write(O/'AFFINE_DIAGNOSIS87_V1.json',dict(utc=utc(),status='DIAGNOSTIC_ONLY_NO_HYPERPARAMETER_CHANGE',rows=rows,source_results_sha256=sha(O/'EVALUATION_RESULTS87_V2.json')))
if __name__=='__main__':main()
