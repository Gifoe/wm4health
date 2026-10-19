"""Independent sealed full-law evaluation of frozen selected weights."""
from query_common44 import *
import datetime,traceback
pilot=read(R/'outputs/pilot_v1/pilot_summary.json');assert pilot['status']=='PASS' and len(pilot['allocation_rows'])==240
training=read(R/'outputs/fits_v1/training_summary.json');assert training['status']=='PASS' and training['fits']==16
O=R/'outputs/quality_v1';assert not O.exists();O.mkdir(parents=True)
state={'status':'running','pid':os.getpid(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'models_completed':0,'models_total':16,'eval_opened':True,'pilot_sha256':sha(R/'outputs/pilot_v1/pilot_summary.json'),'rule_sha256':sha(R/'EVALUATION_RULE.md')};dump(O/'status.json',state)
try:
 rows=[]
 for summary in training['summaries']:
  label=f"{summary['case']}_{summary['family']}_{summary['seed']}";state['current_model']=label;dump(O/'status.json',state);folder=O/label;folder.mkdir();model,module=selected_model(summary);before=checksum(module);observations=np.load(R/'data'/summary['case']/'eval.npz')['observations'];assert observations.shape==(64,LEN,P);samples=[];energies=[];mse=[];coverage=[];durations=[];prefix_times=[];seeds=[]
  for context in range(64):
   history=observations[context:context+1,:H];draw_seed=202645000+context;seeds.append(draw_seed)
   with prefix_cache(summary['family'],model):
    torch.cuda.synchronize();start=time.perf_counter();warm_prefix(summary['family'],model,history);torch.cuda.synchronize();prefix_times.append(time.perf_counter()-start)
    torch.cuda.synchronize();start=time.perf_counter();paths=sample_model(summary['family'],model,history,64,draw_seed);array=paths[0].cpu().numpy();torch.cuda.synchronize();durations.append(time.perf_counter()-start)
   assert np.isfinite(array).all();samples.append(array);energies.append(float(energy(paths,observations[context:context+1,H:])[0]));target=observations[context,H:];mse.append(float(np.mean((array.mean(0)-target)**2)));lo,hi=np.quantile(array,[.05,.95],axis=0);coverage.append(float(np.mean((target>=lo)&(target<=hi))))
   if (context+1)%16==0:print(json.dumps({'model':label,'EVAL_origins_completed':context+1}),flush=True)
  raw=folder/'quality_arrays.npz';np.savez_compressed(raw,paths=np.stack(samples),targets=observations[:,H:],histories=observations[:,:H],draw_seeds=np.asarray(seeds));row={'model':label,'case':summary['case'],'family':summary['family'],'seed':summary['seed'],'joint_energy_mean':float(np.mean(energies)),'joint_energy_by_origin':energies,'mean_prediction_MSE':float(np.mean(mse)),'MSE_by_origin':mse,'marginal_90pct_coverage':float(np.mean(coverage)),'coverage_by_origin':coverage,'prefix_seconds_by_origin':prefix_times,'generation_and_CPU_transfer_seconds_by_origin':durations,'iid_draws_per_origin':64,'origins':64,'checkpoint_sha256':summary['checkpoint_sha256'],'raw_arrays_sha256':sha(raw),'eval_sha256':sha(R/'data'/summary['case']/'eval.npz')};dump(folder/'scores.json',row);rows.append(row);assert checksum(module)==before;state['models_completed']+=1;dump(O/'status.json',state);del model,module;torch.cuda.empty_cache()
 result={'status':'PASS','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'models':16,'rows':rows,'pilot_sha256':state['pilot_sha256'],'evaluation_rule_sha256':state['rule_sha256'],'source_sha256':sha(pathlib.Path(__file__)),'no_selection_on_EVAL':True};dump(O/'quality_summary.json',result);state.update(status='completed',completed_utc=result['utc']);dump(O/'status.json',state)
except Exception:
 state.update(status='failed',traceback=traceback.format_exc());dump(O/'status.json',state);raise
