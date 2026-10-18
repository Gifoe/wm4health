from common63 import *
import numpy as np,math
a=read(OUT/'AUDIT63_V1.json');assert sha(a['traces'])==a['traces_sha256'];tr=np.load(a['traces']);maxerr=0.;checks=[]
for r in a['records']:
 s=str(r['untrained_seed']);loc=tr[s+'_component_location'];sd=tr[s+'_component_scale'];n=loc.shape[0]
 assert n==20 and np.array_equal(tr[s+'_sample20'],loc) and np.isfinite(tr[s+'_sample40']).all() and (sd>0).all()
 mean=np.sum(loc,axis=0)/n;between=np.sum((loc-mean[None])**2,axis=0)/n;within=np.sum(sd*sd,axis=0)/n
 for key,value in [('mean_between_component_variance',between.mean()),('mean_missing_within_component_variance',within.mean()),('mean_full_mixture_variance',(within+between).mean())]:
  err=abs(float(value)-r['sample_count_law'][key]);assert err<1e-10;maxerr=max(maxerr,err)
 for row in r['estimator_interventions']:
  tag=str(row['estimator_weight_multiplier']);rs=tr[s+'_residual_s_'+tag];rf=tr[s+'_residual_f_'+tag];ss=np.mean(1/(1+np.exp(-10*(1e-5-np.abs(rs)))));sf=np.mean(1/(1+np.exp(-10*(1e-5-np.abs(rf)))));aux=.001*(1-ss)+.01*max(.2,ss-sf)
  err=abs(float(aux)-row['auxiliary_penalty']);assert err<1e-12;assert abs(row['native_total_loss']-row['unchanged_predictive_loss']-aux)<1e-10;maxerr=max(maxerr,err)
  if tag=='0.0':assert np.count_nonzero(rs)==np.count_nonzero(rf)==0 and row['predictive_output_max_difference']==0 and row['aux_grad_estimator_norm']==row['aux_grad_encoder_norm']==0
 floor=.001*(1-1/(1+np.exp(-1e-4)))+.002;assert abs(r['estimator_interventions'][-1]['auxiliary_penalty']-floor)<1e-12
 assert sha(r['weight_path'])==r['weights_sha256'];checks.append({'seed':r['untrained_seed'],'mean_missing_variance':float(within.mean()),'zero_head_regularizer_floor':floor,'global_minimum_argument':'sigmoid(10*(delta-abs(e))) <= sigmoid(10*delta), and max(.2,diff)>=.2; zero estimator attains both bounds independently of predictive state.'})
# Existing normalized density guard on a specified invertible residual transform.
guard=[{'a':v,'normalized_gaussian_nll_at_residual_1':.5*(v*v)+.5*math.log(2*math.pi)-math.log(v),'quadratic_term_without_jacobian':.5*v*v} for v in [1.,.1,.01,.001,1e-6]]
assert all(guard[i]['normalized_gaussian_nll_at_residual_1']<guard[i+1]['normalized_gaussian_nll_at_residual_1'] for i in range(len(guard)-1))
dump(OUT/'INDEPENDENT_REPLAY63_V1.json',{'utc':utc(),'status':'PASS','audited_weight_fixtures':len(checks),'trained_models':0,'optimizer_updates':0,'max_arithmetic_difference':maxerr,'checks':checks,'existing_change_of_variables_guard':guard,'zero_transform':'noninvertible and not a finite normalized density transformation','audit_sha256':sha(OUT/'AUDIT63_V1.json'),'scope':'Independent NumPy moment/penalty replay and analytic collapse bound; not an independent reimplementation of the whole neural model or a trained alternative.'})
print(json.dumps({'status':'PASS','fixtures':len(checks),'max_arithmetic_difference':maxerr,'optimizer_updates':0}))
