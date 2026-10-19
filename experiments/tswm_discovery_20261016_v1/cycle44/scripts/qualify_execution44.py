"""VAL-only checks of estimator assembly and independent function arithmetic."""
from query_execution44 import *
from scipy.special import logsumexp
O=R/'outputs/query_execution_qualification_v1';assert not O.exists();O.mkdir(parents=True);training=read(R/'outputs/fits_v1/training_summary.json');assert training['fits']==16
try:
 checks=[]
 for family in ['lgssm','ar_mixture','spf_native','spf_condition_trainable']:
  summary=next(s for s in training['summaries'] if s['case']=='gaussian' and s['family']==family and s['seed']==SEEDS[0]);model,module=selected_model(summary);before=checksum(module);history=np.load(R/'data/gaussian/val.npz')['observations'][:1,:H];spec=read(R/'data/gaussian/functional_spec.json');s=202646800
  with prefix_cache(family,model):
   warm_prefix(family,model,history);paths,g=fine_values(family,model,history,spec,s);anti_paths,gm=fine_values(family,model,history,spec,s,True);parameters=coarse_parameters_ssm(model,history) if family=='lgssm' else None
   x=paths[0].cpu().numpy().astype(np.float64);spatial=x.mean(-1);independent=np.column_stack((spatial.mean(-1),spec['smoothmax_tau']*(logsumexp(spatial/spec['smoothmax_tau'],axis=-1)-math.log(FUT)),np.mean(np.diff(x,axis=1)**2,axis=(1,2)),np.mean(x>spec['point_threshold'],axis=(1,2)),np.max(spatial,axis=1)>spec['spatial_mean_threshold']))/np.asarray(spec['scales']);fn_error=float(np.max(np.abs(g-independent)));assert fn_error<1e-5,(family,fn_error)
   expected={'mc':g.mean(0,dtype=np.float64),'antithetic_mc':((g+gm)/2).mean(0,dtype=np.float64)}
   if family.startswith('spf') or family=='lgssm':
    c=coarse_values(family,model,history,spec,s,parameters,fine=paths);cm=coarse_values(family,model,history,spec,s,parameters,True,anti_paths);ci=coarse_values(family,model,history,spec,s+20000,parameters);cim=coarse_values(family,model,history,spec,s+20000,parameters,True);expected['mlmc']=ci.mean(0,dtype=np.float64)+(g-c).mean(0,dtype=np.float64);expected['antithetic_mlmc']=((ci+cim)/2).mean(0,dtype=np.float64)+((g-c+gm-cm)/2).mean(0,dtype=np.float64)
   if family in ['lgssm','ar_mixture']:
    moments=ssm_query_moments(model,history) if family=='lgssm' else None;analytic=ssm_analytic_readout(moments,spec) if moments is not None else None;rv=readout_values(family,model,history,spec,s,moments,analytic);ra=readout_values(family,model,history,spec,s,moments,analytic,True);expected['conditional_readout']=rv.mean(0,dtype=np.float64);expected['antithetic_conditional_readout']=((rv+ra)/2).mean(0,dtype=np.float64)
    if analytic is not None:expected['analytic_readout']={str(k):float(v) for k,v in analytic.items()}
  errors={}
  for policy,reference in expected.items():
   value,seconds=actual_request(family,model,history,spec,policy,64 if 'mlmc' in policy else 0,0 if policy=='analytic_readout' else 64,s)
   error=max(abs(value[k]-v) for k,v in reference.items()) if isinstance(reference,dict) else float(np.max(np.abs(np.asarray(value)-reference)));assert error<1e-5,(family,policy,error);assert seconds>0 and math.isfinite(seconds);errors[policy]=error
  assert checksum(module)==before;checks.append({'family':family,'independent_numpy_function_max_abs':fn_error,'request_assembly_max_abs':errors});del model,module;torch.cuda.empty_cache()
 allocation_checks=[]
 for row in read(R/'outputs/pilot_v1/pilot_summary.json')['allocation_rows']:
  candidates=row['all_feasible_policies'];chosen=row['selected'];assert chosen in candidates and chosen['predicted_variance']==min(c['predicted_variance'] for c in candidates);assert all(chosen[k]>=0 and chosen[k]%64==0 for k in ['n0','n1']);allocation_checks.append((row['model'],row['function_index'],row['fine_equivalent_budget']))
 assert len(allocation_checks)==240 and len(set(allocation_checks))==240
 result={'status':'PASS','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checks':checks,'allocation_rows_checked':len(allocation_checks),'execution_source_sha256':sha(R/'scripts/query_execution44.py'),'query_common_sha256':sha(R/'scripts/query_common44.py'),'source_sha256':sha(pathlib.Path(__file__)),'numerical_qualification_used_VAL_only':True,'new_training_updates':0};dump(O/'qualification.json',result);print(json.dumps(result),flush=True)
except Exception:
 dump(O/'qualification.json',{'status':'FAIL','traceback':traceback.format_exc(),'numerical_qualification_used_VAL_only':True});raise
