import sys,pathlib,time,collections
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common78_v1 import *
sys.path.insert(0,str(PROJECT/'cycle71'/'scripts'))
from runtime_affinity71_v1 import configure
AFFINITY=configure('efficient')
import numpy as np
import history_alignment78_v1 as align
def g(y,c):
 if c=='identity':return y.copy()
 if c=='affine':return .35+1.4*y
 return .25+1.2*y+.8*np.tanh(1.5*y)
def gi(x,c):
 if c=='identity':return x.copy()
 if c=='affine':return (x-.35)/1.4
 y=(x-.25)/1.2
 for _ in range(40):y-=(g(y,c)-x)/(1.2+1.2*(1-np.tanh(1.5*y)**2))
 return y
def score(p,y):
 e=p-y;finite=bool(np.isfinite(e).all());mse=float(np.mean(e**2)) if finite else None;mae=float(abs(e).mean()) if finite else None
 if finite and not (np.isfinite(mse) and np.isfinite(mae)):finite=False
 return dict(finite=finite,points=int(e.size),mse=mse if finite else None,rmse=float(np.sqrt(mse)) if finite else None,mae=mae if finite else None)
def summarize_parameters(par,meta):
 good=np.isfinite(par).all(axis=1);quantiles=[0,.05,.5,.95,1.]
 return dict(origins=len(par),finite_parameters=int(good.sum()),q_percentiles=np.quantile(par[good,0],quantiles).tolist() if good.any() else None,r_percentiles=np.quantile(par[good,1],quantiles).tolist() if good.any() else None,quantile_levels=quantiles,solver_successes=sum(z['success'] for z in meta),solver_nonconverged=sum(not z['success'] for z in meta),status_counts=dict(collections.Counter(str(z['status']) for z in meta)),total_evaluations=sum(z['evaluations'] for z in meta),any_active_bound_origins=sum(z['active_bounds']>0 for z in meta),cost_mean=float(np.mean([z['cost'] for z in meta])),cost_max=float(max(z['cost'] for z in meta)))
def main():
 started=time.perf_counter();art=storage('artifact')/'history_alignment_v1';art.mkdir(exist_ok=False);s75=read(PROJECT/'cycle75'/'outputs'/'DATA_SPLITS75_V1.json');m74=read(PROJECT/'cycle74'/'outputs'/'TRAIN_DATA74_V2.json');r77=read(PROJECT/'cycle77'/'outputs'/'COMPOSITION_RESULTS77_V2.json');pin77=read(PROJECT/'cycle77'/'outputs'/'SOURCE_PIN77_V2.json');raw={};coefs={};noise={};rows=[];qualification=[]
 for rr in m74['records']:
  assert sha(rr['array_path'])==rr['array_sha256'];a=np.load(rr['array_path'],allow_pickle=False);sc=s75['scaling'][rr['dataset']];raw[rr['case']]=((a['u']-sc['mean'][0])/sc['std'][0],(a['y']-sc['mean'][1])/sc['std'][1])
 for name,cp in s75['source_models'].items():
  assert sha(cp['path'])==cp['sha256'];coef=np.load(cp['path'],allow_pickle=False);coefs[name]=coef;ss=0.;n=0
  for split in s75['splits']:
   if split['dataset']!=name:continue
   u,y=raw[split['case']];end=split['source'][1]
   for start in range(64,end,8192):
    t=np.arange(start,min(start+8192,end));X=np.column_stack([np.ones(len(t)),u[t]]+[u[t-j] for j in range(1,65)]+[y[t-j] for j in range(1,65)]);e=X@coef-y[t];ss+=float(e@e);n+=len(t)
  noise[name]=dict(source_rows=n,source_squared_residual_sum=ss,source_residual_mse=ss/n,innovation_ridge=max(ss/n,1e-12),noise_or_posterior_calibration_claim=False)
 dump(OUT/'SOURCE_PIN78_V1.json',dict(utc=utc(),protocol_sha256=sha(ROOT/'PROTOCOL78_V1.md'),previous_result_sha256=sha(PROJECT/'cycle77'/'outputs'/'COMPOSITION_RESULTS77_V2.json'),previous_pin_sha256=sha(PROJECT/'cycle77'/'outputs'/'SOURCE_PIN77_V2.json'),split_sha256=sha(PROJECT/'cycle75'/'outputs'/'DATA_SPLITS75_V1.json'),source_models=s75['source_models'],source_residual_scales=noise,map_pins=pin77['map_pins'],official_test_values_parsed=False))
 for base in r77['rows']:
  if base['model'] is None:continue
  destination=base['destination'];condition=base['condition'];source=base['calibration_source'];family=base['family'];coef=coefs[destination];model=base['model'];sourcepin=next(z for z in pin77['map_pins'] if z['calibration_source']==source and z['condition']==condition and z['family']==family);assert sha(sourcepin['map_path'])==sourcepin['map_sha256'];assert model==read(sourcepin['map_path']);uh=[];xh=[];fu=[];target=[];canonical=[];ids=[];origins=[]
  for rr in m74['records']:
   if rr['dataset']!=destination:continue
   u,y=raw[rr['case']];x=g(y,condition)
   for o in rr['dev_origins']:uh.append(u[o-128:o]);xh.append(x[o-128:o]);fu.append(u[o:o+128]);target.append(x[o:o+128]);canonical.append(y[o:o+128]);ids.append(rr['case']);origins.append(o)
  uh,xh,fu,target,canonical=map(np.asarray,[uh,xh,fu,target,canonical]);ids=np.asarray(ids);origins=np.asarray(origins);denom=base['normalizer_eval_only_destination_calibration_x_std']
  split=next(z for z in s75['splits'] if z['dataset']==destination);u,y=raw[split['case']];o=split['calibration_fit'][0]+128;qual=align.qualify(coef,model,u[o-128:o][None],g(y[o-128:o],condition)[None]);qualification.append(dict(destination=destination,calibration_source=source,condition=condition,family=family,**qual))
  for method in ['offset_innovation_ridge','affine_innovation_ridge','affine_past_simulation']:
   state('PROGRESS78_V1.json',dict(status='HISTORY_ONLY_NUISANCE_FITS_RUNNING',destination=destination,calibration_source=source,condition=condition,family=family,method=method,closed_rows=len(rows)))
   fit_started=time.perf_counter();parameters,metadata=align.fit_history(coef,model,uh,xh,noise[destination]['innovation_ridge'],method);fit_seconds=time.perf_counter()-fit_started
   # Future input is supplied only after the historical nuisance estimates are complete.
   p=align.forecast(coef,model,uh,xh,fu,parameters);cp=gi(p,condition);perrecord=[]
   for case in dict.fromkeys(ids.tolist()):
    mask=ids==case;meta=[metadata[i] for i in np.flatnonzero(mask)];observed=score(p[mask],target[mask]);perrecord.append(dict(case=case,origins=int(mask.sum()),observed=observed,observed_nrmse=observed['rmse']/denom if observed['finite'] else None,canonical=score(cp[mask],canonical[mask]),nuisance=summarize_parameters(parameters[mask],meta)))
   prefix=destination+'_'+condition+'_'+source+'_'+family+'_'+method;arraypath=art/(prefix+'.npz');np.savez_compressed(arraypath,pred=p,target=target,canonical_pred=cp,canonical_target=canonical,record_ids=ids,origins=origins,parameters=parameters);fitpath=art/(prefix+'_fit_metadata.json');dump(fitpath,dict(destination=destination,condition=condition,source=source,family=family,method=method,origins=origins.tolist(),record_ids=ids.tolist(),metadata=metadata,parameters_path=str(arraypath),parameters_sha256=sha(arraypath)))
   observed=score(p,target);row=dict(destination=destination,calibration_source=source,condition=condition,family=family,method=method,cross_system=source!=destination,source_new_view_calibration_available=source==destination,observed=observed,observed_nrmse=observed['rmse']/denom if observed['finite'] else None,canonical=score(cp,canonical),evaluation_only_x_std=denom,per_record=perrecord,nuisance=summarize_parameters(parameters,metadata),nuisance_fit_seconds=fit_seconds,array_path=str(arraypath),array_sha256=sha(arraypath),fit_metadata_path=str(fitpath),fit_metadata_sha256=sha(fitpath),frozen_baseline_observed_nrmse=base['observed_nrmse'],frozen_baseline_canonical_mse=base['canonical']['mse'],destination_known_sensor_observed_nrmse=next(z['observed_nrmse'] for z in r77['rows'] if z['destination']==destination and z['condition']==condition and z['method']=='known_sensor'));rows.append(row);print(json.dumps({k:row[k] for k in ['destination','calibration_source','condition','family','method','observed_nrmse','nuisance_fit_seconds']}),flush=True)
 result=dict(utc=utc(),status='THREE_ORDINARY_HISTORY_ONLY_ALIGNMENT_RECIPES_DEV_CLOSED',protocol_sha256=sha(ROOT/'PROTOCOL78_V1.md'),source_pin_sha256=sha(OUT/'SOURCE_PIN78_V1.json'),rows=rows,qualification=qualification,nuisance_fits=sum(z['nuisance']['origins'] for z in rows),source_dynamics_refits=0,base_map_refits=0,new_neural_fits=0,new_method=False,future_outputs_available_to_fit=False,future_inputs_available_to_fit=False,new_view_calibration_available_to_cross_arm=False,official_test_values_parsed=False,retrospective_not_independent_confirmation=True,accepted_final_directions=0,overall_goal='ACTIVE_UNACHIEVED',process_affinity=AFFINITY,wall_seconds=time.perf_counter()-started,local_result_files_written=0)
 dump(OUT/'ALIGNMENT_RESULTS78_V1.json',result);state('PROGRESS78_V1.json',dict(status=result['status'],rows=len(rows),nuisance_fits=result['nuisance_fits'],wall_seconds=result['wall_seconds']));print(json.dumps(dict(status=result['status'],rows=len(rows),nuisance_fits=result['nuisance_fits'],wall_seconds=result['wall_seconds'])),flush=True)
if __name__=='__main__':main()
