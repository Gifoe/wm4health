"""Frozen VAL-only estimator allocation. Run after the training process exits."""
from query_common44 import *
import datetime,traceback,subprocess
FITS=R/'outputs/fits_v1';training=read(FITS/'training_summary.json');assert training['status']=='PASS' and training['fits']==16 and not training['eval_opened']
qualification=read(R/'outputs/query_qualification_v1/qualification.json');assert qualification['status']=='PASS' and qualification['query_common_sha256']==sha(R/'scripts/query_common44.py')
O=R/'outputs/pilot_v1';assert not O.exists();O.mkdir(parents=True);state={'status':'running','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'models_completed':0,'models_total':16,'eval_opened':False};dump(O/'status.json',state)
try:
 all_rows=[];parity=[]
 for index,summary in enumerate(training['summaries']):
  calibration_started=time.perf_counter();label=f"{summary['case']}_{summary['family']}_{summary['seed']}";folder=O/label;folder.mkdir();state['current_model']=label;dump(O/'status.json',state);model,module=selected_model(summary);before=checksum(module);val=np.load(R/'data'/summary['case']/'val.npz')['observations'];spec=read(R/'data'/summary['case']/'functional_spec.json');family=summary['family'];history=val[:1,:H]
  plain=sample_model(family,model,history,64,202644800)
  with prefix_cache(family,model):cached=sample_model(family,model,history,64,202644800)
  cache_delta=float((plain-cached).abs().max());assert cache_delta==0,(label,cache_delta);parity.append({'model':label,'cache_forward_max_abs':cache_delta})
  variances=[];time_fine=[];time_coarse=[];setup_times=[];actual_anti=[];raws={};readout_variances=[];readout_costs=[];readout_anti_costs=[];analytic_costs=[];pair_extra_costs=[]
  for context in range(4):
   history=val[context:context+1,:H];fg=[];fminus=[];coarse=[];cminus=[];times_f=[];times_c=[];rvals=[];rminus=[];setup_start=time.perf_counter()
   with prefix_cache(family,model):
    # Populate ordinary deterministic caches; charge their setup separately.
    warm_prefix(family,model,history)
    parameters=coarse_parameters_ssm(model,history) if family=='lgssm' else None
    torch.cuda.synchronize();setup_times.append(time.perf_counter()-setup_start)
    if family=='lgssm':
     torch.cuda.synchronize();start=time.perf_counter();moments=ssm_query_moments(model,history);analytic=ssm_analytic_readout(moments,spec);torch.cuda.synchronize();analytic_costs.append(time.perf_counter()-start)
    for block in range(16):
     s=202644900+context*100+block;torch.cuda.synchronize();start=time.perf_counter();fine=sample_model(family,model,history,64,s);g=function_values(fine,spec);g_array=g[0].cpu().numpy();torch.cuda.synchronize();times_f.append((time.perf_counter()-start)/64)
     torch.cuda.synchronize();start=time.perf_counter();anti=sample_model(family,model,history,64,s,antithetic=True);gm=function_values(anti,spec);gm_array=gm[0].cpu().numpy();torch.cuda.synchronize();actual_anti.append((time.perf_counter()-start)/64);fg.append(g_array);fminus.append(gm_array)
     if family in ['ar_mixture','lgssm']:
      for anti_flag in [False,True]:
       torch.cuda.synchronize();start=time.perf_counter()
       if family=='ar_mixture':pc,eta=draw_ar_pc(model,history,64,s+3000000,anti_flag);readout=ar_conditional_readout(model,pc,spec,eta)
       else:readout=ssm_marginal_readout(moments,analytic,spec,64,s+3000000,anti_flag)
       values=readout.cpu().numpy();torch.cuda.synchronize();elapsed=(time.perf_counter()-start)/64
       if anti_flag:rminus.append(values);readout_anti_costs.append(elapsed)
       else:rvals.append(values);readout_costs.append(elapsed)
     if family.startswith('spf'):
      torch.cuda.synchronize();start=time.perf_counter();c=sample_spf(model,history,64,s,timescale='yearly');fc=function_values(c,spec,coarse=True);fc[0].cpu().numpy();torch.cuda.synchronize();times_c.append((time.perf_counter()-start)/64);cm=sample_spf(model,history,64,s,timescale='yearly',antithetic=True);fcm=function_values(cm,spec,coarse=True)
     elif family=='lgssm':
      # Pair the actual fine aggregate. Independently time the matching direct law.
      torch.cuda.synchronize();start=time.perf_counter();fc=function_values(fine.mean(2,keepdim=True),spec,coarse=True);fc[0].cpu().numpy();torch.cuda.synchronize();pair_extra_costs.append((time.perf_counter()-start)/64);fcm=function_values(anti.mean(2,keepdim=True),spec,coarse=True);torch.cuda.synchronize();start=time.perf_counter();c=draw_ssm_coarse(model,parameters,64,s+1000000);function_values(c,spec,coarse=True).cpu().numpy();torch.cuda.synchronize();times_c.append((time.perf_counter()-start)/64)
     else:continue
     coarse.append(fc[0].cpu().numpy());cminus.append(fcm[0].cpu().numpy())
   g=np.concatenate(fg);gm=np.concatenate(fminus);v_g=np.var(g,0,ddof=1);v_a=np.var((g+gm)/2,0,ddof=1)
   if coarse:
    c=np.concatenate(coarse);cm=np.concatenate(cminus);v0=np.var(c,0,ddof=1);v1=np.var(g-c,0,ddof=1);va0=np.var((c+cm)/2,0,ddof=1);va1=np.var((g+gm-c-cm)/2,0,ddof=1)
   else:c=cm=np.empty((0,5));v0=v1=va0=va1=np.zeros(5)
   variances.append(np.stack((v_g,v_a,v0,v1,va0,va1)));time_fine.extend(times_f);time_coarse.extend(times_c);raws.update({f'g_ctx{context}':g,f'ganti_ctx{context}':gm,f'c_ctx{context}':c,f'canti_ctx{context}':cm})
   if rvals:
    rv=np.concatenate(rvals);rm=np.concatenate(rminus);raws.update({f'readout_ctx{context}':rv,f'readoutanti_ctx{context}':rm});readout_variances.append(np.stack((np.var(rv,0,ddof=1),np.var((rv+rm)/2,0,ddof=1))))
   print(json.dumps({'model':label,'pilot_VAL_context_completed':context}),flush=True)
  np.savez_compressed(folder/'function_draws.npz',**raws);V=np.mean(variances,axis=0);cf=float(np.median(time_fine));cc=float(np.median(time_coarse)) if time_coarse else None;cp=cf+(cc if family.startswith('spf') else float(np.median(pair_extra_costs))) if cc is not None else None;rows=[];vr=np.mean(readout_variances,axis=0) if readout_variances else None;cr=float(np.median(readout_costs)) if readout_costs else None
  for f in range(5):
   for budget in [128,512,2048]:
    chosen,choices=allocate(V[:,f],[cf,cc,cp],budget)
    if vr is not None:
     for anti_flag in [False,True]:
      count=int(budget*cf/(cr*(2 if anti_flag else 1)))//64*64
      if count>=64:choices.append({'policy':'antithetic_conditional_readout' if anti_flag else 'conditional_readout','n0':0,'n1':count,'predicted_variance':float(vr[int(anti_flag),f]/count),'predicted_sampling_seconds':count*cr*(2 if anti_flag else 1)})
    if family=='lgssm' and f in [0,2,3]:choices.append({'policy':'analytic_readout','n0':0,'n1':0,'predicted_variance':0.,'predicted_sampling_seconds':0.,'fixed_query_seconds':float(np.mean(analytic_costs))})
    chosen=min(choices,key=lambda row:row['predicted_variance']);rows.append({'model':label,'function':spec['names'][f],'function_index':f,'fine_equivalent_budget':budget,'selected':chosen,'all_feasible_policies':choices})
  metadata={'model':label,'case':summary['case'],'family':family,'seed':summary['seed'],'pilot_variances':V.tolist(),'fine_seconds_per_draw':cf,'coarse_seconds_per_draw':cc,'pair_seconds_per_unit':cp,'actual_antithetic_seconds_per_draw_median':float(np.median(actual_anti)),'conditional_readout_variance':vr.tolist() if vr is not None else None,'conditional_readout_seconds_per_draw':cr,'conditional_readout_anti_seconds_per_draw':float(np.median(readout_anti_costs)) if readout_anti_costs else None,'analytic_setup_seconds_by_context':analytic_costs,'setup_seconds_by_context':setup_times,'cost_profile':'microbatch64, CUDA synchronized, generation plus all5function evaluation and CPU results transfer','checkpoint_sha256':summary['checkpoint_sha256'],'function_draws_sha256':sha(folder/'function_draws.npz'),'eval_opened':False,'allocation_rule_sha256':sha(R/'ALLOCATION_RULE.md'),'analytic_controls_sha256':sha(R/'ANALYTIC_CONTROLS.md')};dump(folder/'pilot.json',metadata);dump(folder/'allocations.json',rows);all_rows.extend(rows);assert checksum(module)==before
  dump(folder/'calibration_cost.json',{'elapsed_seconds_including_model_load_and_parity':time.perf_counter()-calibration_started,'all5_functions_shared':True,'amortization_requests':[1,100,1000],'not_a_training_cost':True});state['models_completed']+=1;dump(O/'status.json',state);del model,module;torch.cuda.empty_cache()
 result={'status':'PASS','models':16,'allocation_rows':all_rows,'cache_parity':parity,'VAL_histories_per_model':4,'base_draws_per_history':1024,'eval_opened':False,'allocation_rule_sha256':sha(R/'ALLOCATION_RULE.md'),'protocol_sha256':sha(R/'PROTOCOL.md'),'source_sha256':sha(pathlib.Path(__file__))};dump(O/'pilot_summary.json',result);state.update(status='completed',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());dump(O/'status.json',state);print(json.dumps({'status':'completed','models':16,'allocation_rows':len(all_rows),'eval_opened':False}),flush=True)
except Exception:
 state.update(status='failed',traceback=traceback.format_exc());dump(O/'status.json',state);raise
