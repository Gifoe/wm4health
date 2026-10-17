"""Training-only preparation and separately gated scoring of four native EM-option fits."""
from common54 import *
import argparse,pickle
from scipy.optimize import linear_sum_assignment
from evaluate54_v2 import components,affine,r2,propagate
SUP=OUT/'em_option_learning_v1';EVAL=SUP/'evaluation';EVAL.mkdir(exist_ok=True)
def load_seal():
 primary=json.loads((OUT/'SELECTION_SEAL54_V1.json').read_text());extra=json.loads((SUP/'SELECTION_SEAL54_EM_TRUE_V1.json').read_text());assert len(primary['fits'])==12 and len(extra['fits'])==4
 for f in extra['fits']:
  assert f['em'] is True and f['iterations']==10000 and f['original_false_init_arrays_exact_match'] is True
  assert sha(f['checkpoint'])==f['checkpoint_sha256']
 return primary,extra
def prepare():
 source_check();_,extra=load_seal();assert not (OUT/'evaluation54.json').exists(),'All diagnostic maps must precede heldout scoring'
 sealpath=EVAL/'READOUT_SEAL_EM_TRUE54_V1.json';assert not sealpath.exists();records=[]
 cpdir=CKPT/'em_option_learning_v1'
 for f in extra['fits']:
  with np.load(data_path(f['task'])) as ar:tx=ar['train_x'];tz=ar['train_z']
  current=pickle.loads(pathlib.Path(f['initial_checkpoint']).read_bytes());original=pickle.loads((CKPT/(f['matched_false_run']+'_initial.pkl')).read_bytes())
  assert jax.tree.structure(current)==jax.tree.structure(original)
  assert all(np.array_equal(a,b) and np.asarray(a).dtype==np.asarray(b).dtype for a,b in zip(jax.tree.leaves(current),jax.tree.leaves(original)))
  params=pickle.loads(pathlib.Path(f['checkpoint']).read_bytes());c=components('rp',f['task'],f['seed'],params,tx)
  Wz,bz,zi=affine(c['smooth_mean'],tz);Wx,bx,xi=affine(c['smooth_mean'],tx)
  path=cpdir/(f['run']+'_diagnostic_readout.npz');assert not path.exists();np.savez_compressed(path,Wz=Wz,bz=bz,Wx=Wx,bx=bx)
  records.append({'run':f['run'],'task':f['task'],'seed':f['seed'],'file':str(path),'sha256':sha(path),'training_numeric_check':c['check'],'state_readout':zi,'observation_readout':xi,'original_false_initial_parameters_rechecked_exact_match':True})
 dump(sealpath,{'utc':utc(),'status':'ALL_FOUR_EXTRA_TRAIN_ONLY_READOUTS_FROZEN_BEFORE_HELDOUT','readouts':records,'primary_fit_seal_sha256':sha(OUT/'SELECTION_SEAL54_V1.json'),'extra_fit_seal_sha256':sha(SUP/'SELECTION_SEAL54_EM_TRUE_V1.json'),'heldout_evaluation_existed_at_preparation':False})
 print('ALL_FOUR_EM_OPTION_DIAGNOSTIC_MAPS_FROZEN',flush=True)
def evaluate():
 source_check();_,extra=load_seal();readoutseal=EVAL/'READOUT_SEAL_EM_TRUE54_V1.json';rs=json.loads(readoutseal.read_text());main=json.loads((OUT/'evaluation54.json').read_text());mainro=json.loads((OUT/'READOUT_SEAL54_V1.json').read_text())
 assert len(main['records'])==12 and len(mainro['readouts'])==12
 assert rs['utc']<=mainro['utc']<=main['utc'];assert not (EVAL/'summary.json').exists()
 prefix=PLAN['prefix_length'];hs=PLAN['horizons'];records=[];paired=[];checks=[]
 dest=ART/'em_option_learning_v1';dest.mkdir(exist_ok=True)
 for f in extra['fits']:
  task=case(f['task']);K=task['latent_dim'];a=load_data(f['task']);hx=a['heldout_x'];hz=a['heldout_z'];rd=next(r for r in rs['readouts'] if r['run']==f['run']);assert sha(rd['file'])==rd['sha256']
  with np.load(rd['file']) as p:Wz=p['Wz'];bz=p['bz'];Wx=p['Wx'];bx=p['bx']
  with np.load(ART/(f['task']+'_oracle.npz')) as ar:truth=ar['state_truth'];xtruth=ar['observation_truth'];om=ar['state_mean']
  assert np.array_equal(truth,hz[:,[prefix-1+h for h in hs]])
  params=pickle.loads(pathlib.Path(f['checkpoint']).read_bytes());full=components('rp',f['task'],f['seed'],params,hx);ctx=components('rp',f['task'],f['seed'],params,hx[:,:prefix])
  checks.extend([{'run':f['run'],'split':'heldout_full','errors':full['check']},{'run':f['run'],'split':'heldout_prefix','errors':ctx['check']}])
  lm,_=propagate(ctx['smooth_mean'][:,-1],ctx['smooth_cov'][:,-1],ctx['A'],ctx['b'],ctx['Q'],max(hs));pred=lm[:,hs]@Wz+bz;xp=lm[:,hs]@Wx+bx
  ze=((pred-truth)**2).mean(-1);xe=((xp-xtruth)**2).mean(-1);ce=((pred-om)**2).mean(-1);assert all(np.isfinite(v).all() for v in [pred,xp,ze,xe,ce])
  rank=int(np.linalg.matrix_rank(Wz));cond=float(np.linalg.cond(Wz));cond=cond if np.isfinite(cond) else None;Aphys=Wz.T@ctx['A']@np.linalg.inv(Wz.T) if rank==K else None
  ev=np.linalg.eigvals(ctx['A']);tv=np.linalg.eigvals(a['true_A']);i,j=linear_sum_assignment(abs(ev[:,None]-tv[None,:]));ee=float(abs(ev[i]-tv[j]).mean())
  path=dest/(f['run']+'_heldout.npz');assert not path.exists();np.savez_compressed(path,state_mean=pred,auxiliary_observation_mean=xp,state_sequence_mse=ze,observation_sequence_mse=xe,conditional_mean_sequence_mse=ce,full_smoothing_state_mean=full['smooth_mean']@Wz+bz,latent_A=ctx['A'])
  row={'run':f['run'],'method':'rp_emtrue','task':f['task'],'seed':f['seed'],'matched_false_run':f['matched_false_run'],'checkpoint_sha256':f['checkpoint_sha256'],'readout_sha256':rd['sha256'],'artifact':str(path),'artifact_sha256':sha(path),'full_smoothing_R2_train_readout':r2(hz,full['smooth_mean']@Wz+bz),'transition_eigenvalue_mean_absolute_error':ee,'physical_A_relative_frobenius_error':float(np.linalg.norm(Aphys-a['true_A'])/np.linalg.norm(a['true_A'])) if Aphys is not None else None,'state_map_rank':rank,'state_map_condition':cond,'horizons':[{'h':h,'state_mse':float(ze[:,v].mean()),'state_R2':r2(truth[:,v],pred[:,v]),'auxiliary_observation_mse':float(xe[:,v].mean()),'conditional_mean_discrepancy':float(ce[:,v].mean())} for v,h in enumerate(hs)]}
  original=next(r for r in main['records'] if r['run']==f['matched_false_run'])
  paired.append({'task':f['task'],'seed':f['seed'],'true_run':f['run'],'false_run':f['matched_false_run'],'full_smoothing_R2_delta_true_minus_false':row['full_smoothing_R2_train_readout']-original['full_smoothing_R2_train_readout'],'eigenvalue_error_delta_true_minus_false':ee-original['transition_eigenvalue_mean_absolute_error'],'horizons':[{'h':r['h'],**{key+'_delta_true_minus_false':r[key]-next(q[key] for q in original['horizons'] if q['h']==r['h']) for key in ['state_mse','auxiliary_observation_mse','conditional_mean_discrepancy']}} for r in row['horizons']]})
  assert sha(f['checkpoint'])==f['checkpoint_sha256'] and sha(rd['file'])==rd['sha256'];records.append(row);print(json.dumps(row),flush=True)
 dump(EVAL/'summary.json',{'utc':utc(),'status':'COMPLETE_FOUR_MATCHED_NATIVE_EM_OPTION_CONTROLS','records':records,'paired':paired,'all_16_models_and_16_readouts_frozen_before_first_heldout_score':True,'extra_readout_seal_sha256':sha(readoutseal),'primary_evaluation_sha256':sha(OUT/'evaluation54.json'),'independent_numeric_checks':checks,'max_independent_smoother_error':max(max(c['errors'].values()) for c in checks),'protocol_sha256':sha(ROOT/'EM_OPTION_LEARNING54_V1.md'),'evaluation_script_sha256':sha(__file__),'scope':'Existing source-option learning comparison, two tasks and two initialization pairs per task. Not a new method or independent real-domain evidence.'})
 print('EM_OPTION_MATCHED_EVALUATION_COMPLETE',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','evaluate']);a=p.parse_args();prepare() if a.mode=='prepare' else evaluate()
