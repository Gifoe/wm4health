"""Complete frozen-law references, disjoint estimator pools and actual requests."""
from query_common44 import *
import datetime,traceback,subprocess

@torch.no_grad()
def fine_values(family,model,history,spec,s,anti=False):
 paths=sample_model(family,model,history,64,s,antithetic=anti);return paths,function_values(paths,spec)[0].cpu().numpy()

@torch.no_grad()
def coarse_values(family,model,history,spec,s,parameters=None,anti=False,fine=None):
 if family.startswith('spf'):paths=sample_spf(model,history,64,s,timescale='yearly',antithetic=anti)
 elif fine is not None:paths=fine.mean(2,keepdim=True)
 else:paths=draw_ssm_coarse(model,parameters,64,s,antithetic=anti)
 return function_values(paths,spec,coarse=True)[0].cpu().numpy()

@torch.no_grad()
def readout_values(family,model,history,spec,s,moments=None,analytic=None,anti=False):
 if family=='ar_mixture':pc,noise=draw_ar_pc(model,history,64,s,antithetic=anti);values=ar_conditional_readout(model,pc,spec,noise)
 else:values=ssm_marginal_readout(moments,analytic,spec,64,s,antithetic=anti)
 return values.cpu().numpy()

def timed(call):
 torch.cuda.synchronize();start=time.perf_counter();value=call();torch.cuda.synchronize();return value,time.perf_counter()-start

def needed_counts(rows):
 out={'fine':2048,'fine_anti':1024,'paired_coarse':0,'paired_coarse_anti':0,'coarse':0,'coarse_anti':0,'readout':0,'readout_anti':0}
 for row in rows:
  p=row['selected'];name=p['policy'];n0,n1=p['n0'],p['n1']
  if name in ['mlmc','antithetic_mlmc']:
   out['paired_coarse']=max(out['paired_coarse'],n1);out['coarse']=max(out['coarse'],n0)
   if name=='antithetic_mlmc':
    out['paired_coarse_anti']=max(out['paired_coarse_anti'],n1);out['coarse_anti']=max(out['coarse_anti'],n0);out['fine_anti']=max(out['fine_anti'],n1)
  if name in ['conditional_readout','antithetic_conditional_readout']:
   out['readout']=max(out['readout'],n1)
   if name=='antithetic_conditional_readout':out['readout_anti']=max(out['readout_anti'],n1)
 out['fine']=max(out['fine'],out['paired_coarse']);assert all(n%64==0 and n//64<20000 for n in out.values());return out

@torch.no_grad()
def actual_request(family,model,history,spec,policy,n0,n1,draw_seed):
 torch.cuda.synchronize();start=time.perf_counter()
 with prefix_cache(family,model):
  warm_prefix(family,model,history);parameters=coarse_parameters_ssm(model,history) if family=='lgssm' and 'mlmc' in policy else None
  if policy=='analytic_readout' or 'conditional_readout' in policy:
   moments=ssm_query_moments(model,history) if family=='lgssm' else None;analytic=ssm_analytic_readout(moments,spec) if moments is not None else None
  else:moments=analytic=None
  if policy=='analytic_readout':
   estimate={str(k):float(v.cpu()) for k,v in analytic.items()}
  else:
   a=[];b=[]
   for offset in range(0,n1,64):
    s=draw_seed+offset//64
    if 'conditional_readout' in policy:
     value=readout_values(family,model,history,spec,s,moments,analytic)
     if policy=='antithetic_conditional_readout':value=(value+readout_values(family,model,history,spec,s,moments,analytic,True))/2
    else:
     paths,value=fine_values(family,model,history,spec,s)
     if 'mlmc' in policy:value=value-coarse_values(family,model,history,spec,s,parameters,fine=paths)
     if policy.startswith('antithetic'):
      anti_paths,anti_value=fine_values(family,model,history,spec,s,True)
      if 'mlmc' in policy:anti_value=anti_value-coarse_values(family,model,history,spec,s,parameters,True,anti_paths)
      value=(value+anti_value)/2
    a.append(value)
   for offset in range(0,n0,64):
    s=draw_seed+20000+offset//64;value=coarse_values(family,model,history,spec,s,parameters)
    if policy=='antithetic_mlmc':value=(value+coarse_values(family,model,history,spec,s,parameters,True))/2
    b.append(value)
   estimate=np.concatenate(a).mean(0,dtype=np.float64)
   if b:estimate=estimate+np.concatenate(b).mean(0,dtype=np.float64)
   estimate=estimate.tolist()
 torch.cuda.synchronize();return estimate,time.perf_counter()-start

def execute():
 training=read(R/'outputs/fits_v1/training_summary.json');pilot=read(R/'outputs/pilot_v1/pilot_summary.json');quality=read(R/'outputs/quality_verification_v1.json');truth=read(R/'outputs/reference_eval_v1/reference_eval.json');assert training['fits']==16 and pilot['status']=='PASS' and quality['status']=='PASS' and truth['status']=='REFERENCE_READY'
 qualification=read(R/'outputs/query_execution_qualification_v1/qualification.json');assert qualification['status']=='PASS' and qualification['execution_source_sha256']==sha(pathlib.Path(__file__))
 O=R/'outputs/query_execution_v1';assert not O.exists();O.mkdir(parents=True);state={'status':'running','pid':os.getpid(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'models_completed':0,'models_total':16,'current_context':None,'replicates_completed_current_context':0,'eval_opened':True};dump(O/'status.json',state)
 try:
  all_models=[]
  for summary in training['summaries']:
   label=f"{summary['case']}_{summary['family']}_{summary['seed']}";state.update(current_model=label,current_context=None,replicates_completed_current_context=0);dump(O/'status.json',state);folder=O/label;folder.mkdir();allocations=read(R/'outputs/pilot_v1'/label/'allocations.json');assert len(allocations)==15;counts=needed_counts(allocations);family=summary['family'];load_start=time.perf_counter();model,module=selected_model(summary);torch.cuda.synchronize();load_seconds=time.perf_counter()-load_start;before=checksum(module);eval_obs=np.load(R/'data'/summary['case']/'eval.npz')['observations'];spec=read(R/'data'/summary['case']/'functional_spec.json');files=[];requests=[];model_start=time.perf_counter();gpu_context=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
   for context in range(4):
    state.update(current_context=context,replicates_completed_current_context=0);dump(O/'status.json',state);history=eval_obs[context:context+1,:H];context_start=time.perf_counter()
    with prefix_cache(family,model):
     _,prefix_seconds=timed(lambda:warm_prefix(family,model,history));parameters=coarse_parameters_ssm(model,history) if family=='lgssm' else None;moments=ssm_query_moments(model,history) if family=='lgssm' else None;analytic=ssm_analytic_readout(moments,spec) if moments is not None else None;reference=[];reference_start=time.perf_counter()
     for block in range(16384//64):
      s=900000000+context*10000+block
      value=readout_values(family,model,history,spec,s,moments,analytic) if family in ['lgssm','ar_mixture'] else fine_values(family,model,history,spec,s)[1];reference.append(value)
     functions=np.concatenate(reference);reference_seconds=time.perf_counter()-reference_start;reference_mean=functions.mean(0,dtype=np.float64);reference_se=functions.std(0,ddof=1,dtype=np.float64)/math.sqrt(len(functions))
     if analytic is not None:
      for f,value in analytic.items():reference_mean[f]=float(value);reference_se[f]=0.
     ref_file=folder/f'reference_ctx{context}.npz';np.savez_compressed(ref_file,functions=functions,mean=reference_mean,standard_error=reference_se);dump(folder/f'reference_ctx{context}.json',{'context':context,'draws':16384,'kind':'fine_full_paths' if family.startswith('spf') else 'conditional_decoder_noise_integrated' if family=='ar_mixture' else 'exact_functions_0_2_3_and_spatial_mean_Gaussian_for_1_4','mean_scaled':reference_mean.tolist(),'mc_se_scaled':reference_se.tolist(),'generation_seconds':reference_seconds,'raw_sha256':sha(ref_file),'seed_start':900000000+context*10000});files.append({'path':ref_file.name,'sha256':sha(ref_file)})
     for rep in range(16):
      base=300000000+context*10000000+rep*100000;raw={key:[] for key in counts};times={key:[] for key in counts};rep_start=time.perf_counter()
      for offset in range(0,counts['fine'],64):
       s=base+offset//64;(paths,g),seconds=timed(lambda:fine_values(family,model,history,spec,s));raw['fine'].append(g);times['fine'].append(seconds);anti_paths=None
       if offset<counts['fine_anti']:
        (anti_paths,gm),seconds=timed(lambda:fine_values(family,model,history,spec,s,True));raw['fine_anti'].append(gm);times['fine_anti'].append(seconds)
       if offset<counts['paired_coarse']:
        c,seconds=timed(lambda:coarse_values(family,model,history,spec,s,parameters,fine=paths));raw['paired_coarse'].append(c);times['paired_coarse'].append(seconds)
       if offset<counts['paired_coarse_anti']:
        assert anti_paths is not None;c,seconds=timed(lambda:coarse_values(family,model,history,spec,s,parameters,True,anti_paths));raw['paired_coarse_anti'].append(c);times['paired_coarse_anti'].append(seconds)
      for key in ['coarse','coarse_anti','readout','readout_anti']:
       anti=key.endswith('_anti');kind_offset=20000 if key.startswith('coarse') else 40000
       for offset in range(0,counts[key],64):
        s=base+kind_offset+offset//64
        call=(lambda:coarse_values(family,model,history,spec,s,parameters,anti)) if key.startswith('coarse') else (lambda:readout_values(family,model,history,spec,s,moments,analytic,anti));value,seconds=timed(call);raw[key].append(value);times[key].append(seconds)
      raw={key:np.concatenate(values) if values else np.empty((0,5),dtype=np.float32) for key,values in raw.items()};assert all(values.shape==(counts[key],5) and np.isfinite(values).all() for key,values in raw.items());raw_file=folder/f'pool_ctx{context}_rep{rep}.npz';np.savez_compressed(raw_file,**raw);metadata={'context':context,'rep':rep,'base_seed':base,'seed_recipe':'fine/pair base+block; independent coarse base+20000+block; conditional readout base+40000+block; antithetic shares base','microbatch':64,'counts':counts,'actual_constituent_call_seconds':times,'elapsed_generation_and_storage_seconds':time.perf_counter()-rep_start,'raw_sha256':sha(raw_file)};dump(folder/f'pool_ctx{context}_rep{rep}.json',metadata);files.append({'path':raw_file.name,'sha256':sha(raw_file)});state.update(replicates_completed_current_context=rep+1);dump(O/'status.json',state);print(json.dumps({'model':label,'context':context,'replicate_completed':rep+1}),flush=True)
    if context==0:
     configs={('mc',0,b) for b in [128,512,2048]}|{('antithetic_mc',0,b//2) for b in [128,512,2048]}|{(r['selected']['policy'],r['selected']['n0'],r['selected']['n1']) for r in allocations}
     if family=='lgssm':configs.add(('analytic_readout',0,0))
     for index,(policy,n0,n1) in enumerate(sorted(configs)):
      request_seed=800000000+index*100000;estimate,seconds=actual_request(family,model,history,spec,policy,n0,n1,request_seed);requests.append({'context':0,'policy':policy,'n0':n0,'n1':n1,'actual_complete_seconds':seconds,'draw_seed':request_seed,'computed_all_functions':True,'estimate_for_timing_only':estimate})
     dump(folder/'actual_complete_requests.json',requests)
    dump(folder/f'context{context}_cost.json',{'prefix_cache_setup_seconds':prefix_seconds,'elapsed_context_seconds_including_pools_reference_storage_and_timing':time.perf_counter()-context_start});assert checksum(module)==before
   result={'model':label,'case':summary['case'],'family':family,'seed':summary['seed'],'pool_counts':counts,'replicates_per_history':16,'histories':4,'independent_reference_draws':16384,'checkpoint_sha256':summary['checkpoint_sha256'],'allocations_sha256':sha(R/'outputs/pilot_v1'/label/'allocations.json'),'model_load_seconds':load_seconds,'experiment_seconds_excluding_load':time.perf_counter()-model_start,'gpu_context_before':gpu_context,'files':files,'actual_complete_requests':len(requests),'unchanged_selected_weights':True};dump(folder/'execution.json',result);all_models.append(result);state['models_completed']+=1;dump(O/'status.json',state);del model,module;torch.cuda.empty_cache()
  result={'status':'PASS','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'models':16,'executions':all_models,'source_sha256':sha(pathlib.Path(__file__)),'query_common_sha256':sha(R/'scripts/query_common44.py'),'pilot_sha256':sha(R/'outputs/pilot_v1/pilot_summary.json'),'rule_sha256':sha(R/'QUERY_EXECUTION.md'),'eval_opened':True,'no_new_training':True};dump(O/'execution_summary.json',result);state.update(status='completed',completed_utc=result['utc']);dump(O/'status.json',state)
 except Exception:
  state.update(status='failed',traceback=traceback.format_exc());dump(O/'status.json',state);raise

if __name__=='__main__':execute()
