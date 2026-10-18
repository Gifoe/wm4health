from common87_v1 import *
from ordinary87_v1 import identify,FAMILIES
from evaluate87_v2 import metrics
from scipy.linalg import expm
import numpy as np,copy
def main():
 old=read(O/'EVALUATION_RESULTS87_V2.json');new=copy.deepcopy(old);choice=read(O/'ORDINARY_SELECTION87_V1.json')['selected'];assert choice['degree']==1;folder=RAW/'evaluation_v3_affine';folder.mkdir(exist_ok=False);changes=[]
 for cohort in ['retained','crossed','unselected']:
  dd=[]
  for fam in FAMILIES:
   with np.load(RAW/'evaluation_v2'/f'{fam}_{cohort}_v1.npz') as z:dd.append({k:z[k] for k in z.files})
  context=np.concatenate([d['context'] for d in dd]);initial=np.concatenate([d['query_initial'] for d in dd]);coef=identify(context,1,choice['ridge']);aug=np.zeros((256,3,3));aug[:,:2,:2]=np.stack([coef[:,2,:],coef[:,1,:]],axis=-1);aug[:,:2,2]=coef[:,0,:];transition=np.stack([expm(a*.0125) for a in aug]);state=np.concatenate([initial,np.ones((256,1))],axis=1);active=np.ones(256,bool);parts=[state[:,:2].copy()]
  for step in range(800):
   newstate=np.einsum('bij,bj->bi',transition,state);active=active&np.isfinite(newstate).all(-1)&(abs(newstate[:,:2])<1000).all(-1);state=np.where(active[:,None],newstate,np.array([0.,0.,1.]))
   if (step+1)%4==0:parts.append(np.where(active[:,None],state[:,:2],np.nan))
  prediction=np.stack(parts,axis=1);oldrow=next(r for r in old['rows'] if r['model']=='ordinary_weak_polynomial' and r['cohort']==cohort)
  with np.load(oldrow['array_path']) as z:
   oldprediction=z['prediction'];f1=z['context_field_prediction'];f2=z['future_field_prediction'];joint=np.isfinite(prediction)&np.isfinite(oldprediction);delta=float(abs(prediction[joint]-oldprediction[joint]).max())
  out=folder/f'ordinary_{cohort}_exact_v1.npz'
  with out.open('xb') as f:np.savez_compressed(f,prediction=prediction,context_field_prediction=f1,future_field_prediction=f2)
  for fi,fam in enumerate(FAMILIES):
   row=next(r for r in new['rows'] if r['model']=='ordinary_weak_polynomial' and r['cohort']==cohort and r['family']==fam);sl=slice(fi*128,(fi+1)*128);row['horizons']=[metrics(prediction[sl],dd[fi]['truth'],h) for h in [1,5,10]];row['array_path']=str(out);row['array_sha256']=sha(out)
  changes.append(dict(cohort=cohort,original_array_sha256=oldrow['array_sha256'],exact_array_sha256=sha(out),finite_common_max_abs_delta=delta,old_h10_failures=int((~np.isfinite(oldprediction).all(axis=(1,2))).sum()),new_h10_failures=int((~np.isfinite(prediction).all(axis=(1,2))).sum())))
 new['utc']=utc();new['status']='FOURTEEN_MODELS_CLOSED_AFFINE_FLOW_EXACT_REPLAY_PENDING';new['original_eval_results_sha256']=sha(O/'EVALUATION_RESULTS87_V2.json');new['numerical_revision']=dict(only_changed_model='ordinary_weak_polynomial',changes=changes,learning_and_selection_unchanged=True,all_other_prediction_arrays_unchanged=True,note_sha256=sha(C/'NUMERICAL_REVISION87_V1.md'))
 write(O/'EVALUATION_RESULTS87_V3.json',new);write(O/'AFFINE_CORRECTION87_V1.json',dict(utc=utc(),status='ONLY_AFFINE_NUMERICAL_FLOW_REPLACED_NO_HYPERPARAMETER_CHANGE',original_results_sha256=sha(O/'EVALUATION_RESULTS87_V2.json'),revised_results_sha256=sha(O/'EVALUATION_RESULTS87_V3.json'),changes=changes));log(dict(stage='affine_exact_correction',changes=changes))
if __name__=='__main__':main()
