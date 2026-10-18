import sys,pathlib,time
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common76_v1 import *
sys.path.insert(0,str(PROJECT/'cycle71'/'scripts'))
from runtime_affinity71_v1 import configure
AFFINITY=configure('efficient')
import numpy as np
import simulation_calibration76_v1 as sim
def sensor(y,condition):
 if condition=='identity':return y.copy()
 if condition=='affine':return .35+1.4*y
 return .25+1.2*y+.8*np.tanh(1.5*y)
def windows(records,key):
 columns=[[],[],[],[]];ids=[];origins=[]
 for r in records:
  for o in r[key]:
   for out,v in zip(columns,[r['u'][o-128:o],r['x'][o-128:o],r['u'][o:o+128],r['x'][o:o+128]]):out.append(v)
   ids.append(r['case']);origins.append(o)
 return tuple(np.asarray(v) for v in columns),np.asarray(ids),np.asarray(origins)
def metrics(p,y,ids,scale):
 records=[];finite=True
 for case in dict.fromkeys(ids.tolist()):
  ix=ids==case;e=p[ix]-y[ix];good=bool(np.isfinite(e).all());finite &=good
  mse=float(np.mean(e**2)) if good else None;mae=float(abs(e).mean()) if good else None
  if good and not (np.isfinite(mse) and np.isfinite(mae)):good=False;finite=False
  records.append(dict(case=case,origins=int(ix.sum()),forecast_points=int(ix.sum()*128),finite=good,rmse=float(np.sqrt(mse)) if good else None,nrmse=float(np.sqrt(mse)/scale) if good else None,mae=mae if good else None))
 e=p-y
 return dict(per_record=records,all_records_finite=finite,failed_records=sum(not r['finite'] for r in records),rmse=float(np.sqrt(np.mean(e**2))) if finite else None,nrmse=float(np.sqrt(np.mean(e**2))/scale) if finite else None,mae=float(abs(e).mean()) if finite else None)
def main():
 started=time.perf_counter();art=storage('artifact')/'simulation_v1';art.mkdir(exist_ok=False);s75=read(PROJECT/'cycle75'/'outputs'/'DATA_SPLITS75_V1.json');r75=read(PROJECT/'cycle75'/'outputs'/'CALIBRATION_RESULTS75_V1.json');m74=read(PROJECT/'cycle74'/'outputs'/'TRAIN_DATA74_V2.json');rows=[];fits=[];qualifications=[];origins_record=[];raw={}
 for r in m74['records']:
  assert sha(r['array_path'])==r['array_sha256'];a=np.load(r['array_path'],allow_pickle=False);s=s75['scaling'][r['dataset']];raw[r['case']]=dict(u=(a['u']-s['mean'][0])/s['std'][0],y=(a['y']-s['mean'][1])/s['std'][1])
 dump(OUT/'SOURCE_PIN76_V1.json',dict(utc=utc(),protocol_sha256=sha(ROOT/'PROTOCOL76_V1.md'),split_sha256=sha(PROJECT/'cycle75'/'outputs'/'DATA_SPLITS75_V1.json'),previous_results_sha256=sha(PROJECT/'cycle75'/'outputs'/'CALIBRATION_RESULTS75_V1.json'),previous_verified_sha256=sha(PROJECT/'cycle75'/'outputs'/'VERIFICATION75_V1.json'),source_models=s75['source_models'],official_test_numeric_values_parsed=False))
 for dataset in ['EMPS','F16']:
  cp=s75['source_models'][dataset];assert sha(cp['path'])==cp['sha256'];coef=np.load(cp['path'],allow_pickle=False)
  for condition in ['identity','affine','nonlinear']:
   state('PROGRESS76_V1.json',dict(status='CALIBRATION_FIT_RUNNING',dataset=dataset,condition=condition,closed_rows=len(rows)))
   records=[]
   for rr in s75['splits']:
    if rr['dataset']!=dataset:continue
    r=dict(rr,**raw[rr['case']]);r['x']=sensor(r.pop('y'),condition);start,end=r['calibration_fit'];r['fit_origins']=list(range(start+128,end-128+1,128));assert r['fit_origins'];records.append(r)
    origins_record.append(dict(dataset=dataset,condition=condition,case=r['case'],fit_origins=r['fit_origins'],internal_origins=r['internal_origins'],dev_origins=r['dev_origins']))
   train,train_ids,train_origins=windows(records,'fit_origins');valid,valid_ids,valid_origins=windows(records,'internal_origins');scale=float(np.concatenate([r['x'][r['calibration_fit'][0]:r['calibration_fit'][1]] for r in records]).std());oldrow=next(r for r in r75['rows'] if r['dataset']==dataset and r['condition']==condition);assert scale==oldrow['normalizer_calibration_x_std']
   oldmap=next(a for a in r75['adapter_selection'] if a['dataset']==dataset and a['condition']==condition and a['kind']=='monotone');assert sha(oldmap['path'])==oldmap['sha256'];old=read(oldmap['path']);k=np.asarray(old['knots']);beta=np.asarray(old['beta']);q=sim.qualify(coef,k,beta,*train[:3]);qualifications.append(dict(dataset=dataset,condition=condition,**q));print(json.dumps(dict(status='CALIBRATION_NUMERICS_QUALIFIED',dataset=dataset,condition=condition,**q)),flush=True)
   affine=sim.affine_fit(coef,*train);ar,ad=sim.pack(coef,np.asarray(affine['knots']),*valid[:3]);ap=sim.prediction(ar,ad,np.asarray(affine['knots']),np.asarray(affine['beta']));affine['internal_validation_normalized_mse']=float(np.mean(((ap-valid[3])/scale)**2))
   # Represent the learned affine inverse in the fixed17-knot family, without paired labels.
   a=affine['beta'][1];b=affine['beta'][0]-a*affine['knots'][0];affine_initial=np.r_[a*k[0]+b,np.full(len(k)-1,a)]
   def progress(obj):
    state('OPTIMIZER_PROGRESS76_V1.json',dict(dataset=dataset,condition=condition,**obj));print(json.dumps(dict(dataset=dataset,condition=condition,**obj)),flush=True)
   monotone,attempts=sim.monotone_fit(coef,k,train,valid,[('global_affine',affine_initial),('C75_innovation',beta)],scale,progress)
   fitting=dict(dataset=dataset,condition=condition,fit_origins=len(train_origins),fit_target_points=int(train[3].size),internal_origins=len(valid_origins),normalizer_calibration_x_std=scale,affine=affine,monotone_selected=monotone,monotone_optimizer_attempts=attempts);fits.append(fitting);dump(OUT/(dataset+'_'+condition+'_FIT76_V1.json'),fitting)
   # DEV arrays are built only after calibration selection is closed.
   dev,ids,origins=windows(records,'dev_origins');models=[('simulation_affine_global',affine),('simulation_monotone_selected',monotone)]
   for attempt in attempts:models.append(('simulation_monotone_terminal_'+attempt['initializer'],dict(kind='monotone_terminal',knots=k.tolist(),beta=attempt['terminal_beta'],optimizer_success=attempt['optimizer_success'],optimizer_status=attempt['optimizer_status'])))
   for method,model in models:
    kk=np.asarray(model['knots']);bb=np.asarray(model['beta']);R,d=sim.pack(coef,kk,*dev[:3]);p=sim.prediction(R,d,kk,bb);probe=np.linspace(kk[0]-(kk[-1]-kk[0]),kk[-1]+(kk[-1]-kk[0]),1024);rt=float(abs(sim.forward_map(sim.inverse_map(probe,kk,bb),kk,bb)-probe).max());assert rt<1e-8
    mp=art/(dataset+'_'+condition+'_'+method+'_map.json');dump(mp,model);dest=art/(dataset+'_'+condition+'_'+method+'_dev.npz');np.savez_compressed(dest,pred=p,target=dev[3],record_ids=ids,origins=origins)
    row=dict(dataset=dataset,condition=condition,method=method,**metrics(p,dev[3],ids,scale),normalizer_calibration_x_std=scale,map_path=str(mp),map_sha256=sha(mp),array_path=str(dest),array_sha256=sha(dest),roundtrip_max_abs=rt,source_model_sha256=cp['sha256']);rows.append(row);print(json.dumps({key:row[key] for key in ['dataset','condition','method','nrmse','failed_records']}),flush=True)
   state('PROGRESS76_V1.json',dict(status='CONDITION_CLOSED',dataset=dataset,condition=condition,closed_rows=len(rows)))
 result=dict(utc=utc(),status='STANDARD_SIMULATION_ERROR_CALIBRATION_DEV_CLOSED',protocol_sha256=sha(ROOT/'PROTOCOL76_V1.md'),source_pin_sha256=sha(OUT/'SOURCE_PIN76_V1.json'),qualifications=qualifications,fits=fits,origin_contracts=origins_record,rows=rows,new_affine_fits=6,new_monotone_optimizations=12,new_neural_fits=0,new_method=False,official_test_values_parsed=False,constructed_observation_changes=True,changed_physical_dynamics=False,accepted_final_directions=0,overall_goal='ACTIVE_UNACHIEVED',wall_seconds=time.perf_counter()-started,process_affinity=AFFINITY,local_result_files_written=0)
 dump(OUT/'SIMULATION_RESULTS76_V1.json',result);state('PROGRESS76_V1.json',dict(status=result['status'],closed_rows=len(rows),wall_seconds=result['wall_seconds']));print(json.dumps(dict(status=result['status'],rows=len(rows),wall_seconds=result['wall_seconds'])),flush=True)
if __name__=='__main__':main()
