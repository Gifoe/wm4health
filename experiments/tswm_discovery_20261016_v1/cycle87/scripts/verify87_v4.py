from native87_v1 import *
from physics87_v1 import *
from train87_v1 import dataset,validation
from ordinary87_v1 import identify,predict as ordinary_field
from empirical87_v1 import load_prior,weak_design,posterior
from scipy.integrate import solve_ivp
from scipy.linalg import expm
import time,gc
def close(a,b,atol=2e-7,rtol=2e-5):
 if a is None or b is None:assert a is None and b is None,(a,b)
 else:assert np.isclose(a,b,atol=atol,rtol=rtol),(a,b)
def independent_metrics(p,y,h,row):
 n=round(h*20)+1;p=np.asarray(p[:,1:n],float);y=np.asarray(y[:,1:n],float);finite=np.isfinite(p).all(axis=(1,2));v=np.full(len(p),np.nan);v[finite]=np.mean((p[finite]-y[finite])**2,axis=(1,2));r=np.sqrt(v)/np.maximum(np.sqrt(np.mean(y*y,axis=(1,2))),.1)
 assert row['failed_count']==int((~finite).sum()) and row['complete_count']==int(finite.sum())
 close(row['population_mse'],float(v.mean()) if finite.all() else None);close(row['population_mean_nrmse'],float(r.mean()) if finite.all() else None);close(row['population_median_nrmse'],float(np.median(r)) if finite.all() else None)
 close(row['finite_only_mse'],float(v[finite].mean()) if finite.any() else None);close(row['finite_only_mean_nrmse'],float(r[finite].mean()) if finite.any() else None);close(row['failure_penalized_mean_nrmse'],float(np.where(finite,r,1000.).mean()))
def direct_bayes(entries,h,y,noise):
 means=[];logw=[]
 for entry in entries:
  mu=entry['mean'];cov=entry['cov'];s=h@cov@h.T+noise**2*np.eye(len(y));res=y-h@mu;sol=np.linalg.solve(s,res);means.append(mu+cov@h.T@sol);logw.append(np.log(entry['weight'])-.5*(res@sol+np.linalg.slogdet(s)[1]+len(y)*np.log(2*np.pi)))
 weights=np.exp(np.array(logw)-max(logw));weights/=weights.sum();return weights@np.stack(means)
def main():
 start=time.perf_counter();assert read(O/'evaluate_v2/WORKER_EXIT87_V2.json')['exit_code']==0;result=read(O/'EVALUATION_RESULTS87_V3.json');selection=read(O/'SELECTION87_V1.json');ordinary=read(O/'ORDINARY_SELECTION87_V1.json');empirical=read(O/'EMPIRICAL_SELECTION87_V1.json');manifest=read(O/'EVAL_MANIFEST87_V2.json');folder=RAW/'evaluation_v2';check_native()
 assert len(result['rows'])==84 and len(selection['fits'])==9;assert sha(O/'SELECTION87_V1.json')==result['selection_sha256'];assert sha(O/'EVAL_MANIFEST87_V2.json')==result['evaluation_manifest_sha256'];assert sha(O/'EMPIRICAL_SELECTION87_V1.json')==result['empirical_selection_sha256']
 for row in manifest['rows']:assert sha(row['array_path'])==row['array_sha256']
 # Independently validate fixed-marginal pairing and a new high-accuracy physical reference.
 max_physics=0.;physics_scalars=0;cohorts={}
 for fam in FAMILIES:
  dd={}
  for name in ['retained','crossed','unselected']:
   with np.load(folder/f'{fam}_{name}_v1.npz') as z:dd[name]={k:z[k] for k in z.files}
   d=dd[name]
   for k in [0,17,63,127]:
    sol=solve_ivp(lambda t,x:field(fam,d['parameters'][k:k+1],x.reshape(1,1,2)).ravel(),(0,10),d['query_initial'][k],method='DOP853',t_eval=np.arange(201)*.05,rtol=2e-11,atol=2e-12);assert sol.success;diff=float(abs(sol.y.T-d['truth'][k]).max());assert diff<1e-4,(fam,name,k,diff);max_physics=max(max_physics,diff);physics_scalars+=402
  with np.load(folder/f'{fam}_pool_v1.npz') as pop:permutation=pop['context_pair_permutation'];assert np.array_equal(dd['crossed']['context_initial'],dd['retained']['context_initial'][permutation])
  for key in ['parameters','mechanism_ids','query_initial','noise']:assert np.array_equal(dd['retained'][key],dd['crossed'][key]),key
  cohorts[fam]=dd
 combined={name:{key:np.concatenate([cohorts[f][name][key] for f in FAMILIES]) for key in cohorts[FAMILIES[0]][name]} for name in ['retained','crossed','unselected']}
 # Replay every checkpoint's source VAL selection. Initial weights and paired RNG streams are checked explicitly.
 val=dataset('val','unselected');m=load_native();m.eval()
 for p in m.parameters():p.requires_grad_(False)
 initial={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};rng_receipts={};valmax=0.;checkpoints=0
 for fit in selection['fits']:
  assert fit['updates']==3000 and len(fit['trace'])==7;chosen=min(fit['trace'],key=lambda z:(z['val_normalized_field_l1'],z['step']));assert fit['selected']==chosen
  for row in fit['trace']:
   assert sha(row['checkpoint_path'])==row['checkpoint_sha256'];cp=torch.load(row['checkpoint_path'],map_location='cpu',weights_only=False);assert cp['step']==row['step'];m.load_state_dict(cp['state_dict']);replay=validation(m,val);delta=abs(replay-row['val_normalized_field_l1']);assert np.isclose(replay,row['val_normalized_field_l1'],atol=3e-4,rtol=2e-5);valmax=max(valmax,delta);checkpoints+=1
   if row['step']==0:
    for key in initial:assert torch.equal(initial[key],cp['state_dict'][key]),key
   stamp=(fit['seed'],row['step']);state=(cp['batch_rng'],cp['torch_rng'],cp['cuda_rng'])
   if stamp in rng_receipts:
    ref=rng_receipts[stamp];assert ref[0]==state[0] and torch.equal(ref[1],state[1]) and all(torch.equal(a,b) for a,b in zip(ref[2],state[2]))
   else:rng_receipts[stamp]=state
   del cp
  log(dict(stage='verify_all_checkpoint_val',seed=fit['seed'],arm=fit['arm'],closed_checkpoints=checkpoints))
 # Replay native field predictions through direct model_forward, not cached evaluator calls.
 model_specs=[('native_frozen',None)]+[(f"native_seed{f['seed']}_{f['arm']}",f['selected']['checkpoint_path']) for f in selection['fits']];field_scalars=0;field_max=0.
 for name,path in model_specs:
  if path:
   cp=torch.load(path,map_location='cpu',weights_only=False);m.load_state_dict(cp['state_dict']);del cp
  else:m.load_state_dict(initial)
  for cohort,d in combined.items():
   path=folder/'predictions'/f'{name}_{cohort}_v1.npz'
   with np.load(path) as saved:
    for loc_key,pred_key in [('context_locations','context_field_prediction'),('future_locations','future_field_prediction')]:
     replay=predict_field(m,d['context'],d[loc_key]);expected=saved[pred_key];assert np.allclose(replay,expected,atol=3e-4,rtol=3e-5);field_max=max(field_max,float(abs(replay-expected).max()));field_scalars+=expected.size
  log(dict(stage='verify_direct_native_fields',model=name,closed_field_scalars=field_scalars))
 del m;gc.collect();torch.cuda.empty_cache()
 # Check empirical posterior with an independent observation-space Gaussian solve.
 bayesmax=0.
 for choice in empirical['selections']:
  assert sha(choice['prior_path'])==choice['prior_sha256'];entries=load_prior(choice['prior_path']);h,y=weak_design(combined['unselected']['context'][:8]);fast=posterior(entries,h,y,choice['weak_noise_std']).reshape(8,20)
  for k in range(8):
   slow=direct_bayes(entries,h[k],y[k],choice['weak_noise_std']);err=float(abs(slow-fast[k]).max());assert np.allclose(slow,fast[k],atol=2e-5,rtol=2e-5);bayesmax=max(bayesmax,err)
 # Audit all reported numbers and their family/cohort membership, retaining failures.
 metric_cases=0;prediction_scalars=0;affine_max=0.
 for row in result['rows']:
  assert sha(row['array_path'])==row['array_sha256'];d=cohorts[row['family']][row['cohort']];sl=slice(FAMILIES.index(row['family'])*128,(FAMILIES.index(row['family'])+1)*128)
  with np.load(row['array_path']) as z:
   p=z['prediction'][sl];f1=z['context_field_prediction'][sl];f2=z['future_field_prediction'][sl];prediction_scalars+=p.size
   for metrics_row in row['horizons']:independent_metrics(p,d['truth'],metrics_row['horizon'],metrics_row);metric_cases+=1
   close(row['context_field_mse'],float(np.mean((f1.astype(float)-d['context_field'])**2)),atol=1e-6);close(row['future_field_mse'],float(np.mean((f2.astype(float)-d['future_field'])**2)),atol=1e-6)
  if row['model']=='ordinary_weak_polynomial' and ordinary['selected']['degree']==1:
   coef=identify(d['context'],1,ordinary['selected']['ridge']);aug=np.zeros((128,3,3));aug[:,:2,:2]=np.stack([coef[:,2,:],coef[:,1,:]],axis=-1);aug[:,:2,2]=coef[:,0,:];transition=np.stack([expm(a*.05) for a in aug]);state=np.concatenate([d['query_initial'],np.ones((128,1))],axis=1);parts=[state[:,:2].copy()]
   for _ in range(200):state=np.einsum('bij,bj->bi',transition,state);parts.append(state[:,:2].copy())
   exact=np.stack(parts,axis=1);ok=np.isfinite(p).all(axis=(1,2));delta=float((abs(exact[ok]-p[ok])/np.maximum(abs(exact[ok]),1)).max()) if ok.any() else 0.;assert delta<1e-9;affine_max=max(affine_max,delta)
 write(O/'VERIFICATION87_V4.json',dict(utc=utc(),status='SOURCE_FIELDS_SELECTIONS_METRICS_AFFINE_REPLAYED_ROLLOUT_QUALIFICATION_INCOMPLETE',closed_source_checkpoints=checkpoints,max_val_l1_replay_abs=valmax,initial_tensor_equality_and_paired_rng=True,physical_reference_scalars=physics_scalars,max_physical_dop853_abs=max_physics,direct_native_field_scalars=field_scalars,max_direct_native_field_abs=field_max,independent_bayes_max_abs=bayesmax,reported_metric_rows_checked=metric_cases,prediction_scalars_counted=prediction_scalars,max_ordinary_affine_expm_scaled_abs=affine_max,supplemental_rollout_qualification_sha256=sha(O/'NUMERICAL_QUALIFICATION87_V1.json'),supplemental_rollout_qualification_passed=False,complete_native_rollout_independent_reintegration=False,verification_scope='All source selections, direct native fields, physical truth, all reported metrics and exact affine solver; supplementary learned rollout qualification failed and remains unresolved',results_sha256=sha(O/'EVALUATION_RESULTS87_V3.json'),source_sha256=sha(__file__),wall_seconds=time.perf_counter()-start))
 log(dict(stage='verification_complete',checkpoints=checkpoints,native_field_scalars=field_scalars,metric_rows=metric_cases,accepted_directions=0))
if __name__=='__main__':main()
