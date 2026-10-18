import sys,pathlib,time,traceback
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common75_v1 import *
sys.path.insert(0,str(PROJECT/'cycle71'/'scripts'))
from runtime_affinity71_v1 import configure
AFFINITY=configure('efficient')
import numpy as np
import learners75_v1 as learn

def sensor(y,condition):
 if condition=='identity':return y.copy()
 if condition=='affine':return .35+1.4*y
 assert condition=='nonlinear';return .25+1.2*y+.8*np.tanh(1.5*y)
def privileged_inverse(x,condition):
 if condition=='identity':return x.copy()
 if condition=='affine':return (x-.35)/1.4
 lo=(x-.25-.8)/1.2-1.;hi=(x-.25+.8)/1.2+1.
 for _ in range(60):
  mid=(lo+hi)/2;above=sensor(mid,'nonlinear')>x;hi=np.where(above,mid,hi);lo=np.where(above,lo,mid)
 return (lo+hi)/2
def windows(records,key):
 uh=[];xh=[];fu=[];target=[];ids=[];origins=[]
 for r in records:
  for o in r[key]:
   uh.append(r['u'][o-128:o]);xh.append(r['x'][o-128:o]);fu.append(r['u'][o:o+128]);target.append(r['x'][o:o+128]);ids.append(r['case']);origins.append(o)
 return tuple(np.asarray(v) for v in [uh,xh,fu,target]),np.asarray(ids),np.asarray(origins)
def metrics(p,y,ids,scale):
 rows=[];allgood=True
 for case in dict.fromkeys(ids.tolist()):
  ix=ids==case;e=p[ix]-y[ix];finite=bool(np.isfinite(e).all());allgood &=finite
  if finite:
   with np.errstate(over='ignore'):mse=float(np.mean(e**2));mae=float(np.mean(abs(e)))
   finite=bool(np.isfinite(mse) and np.isfinite(mae));allgood &=finite
  rows.append(dict(case=case,origins=int(ix.sum()),forecast_points=int(ix.sum()*128),finite=finite,rmse=float(np.sqrt(mse)) if finite else None,nrmse=float(np.sqrt(mse)/scale) if finite else None,mae=mae if finite else None))
 if allgood:e=p-y;rmse=float(np.sqrt(np.mean(e**2)));mae=float(np.mean(abs(e)))
 return dict(per_record=rows,all_records_finite=allgood,failed_records=sum(not r['finite'] for r in rows),rmse=rmse if allgood else None,nrmse=rmse/scale if allgood else None,mae=mae if allgood else None)
def main():
 started=time.perf_counter();art=storage('artifact')/'experiment_v1';art.mkdir(exist_ok=False);manifest=read(PROJECT/'cycle74'/'outputs'/'TRAIN_DATA74_V2.json');prepared={};splitrows=[];scaling={};models={};modelmeta={};rows=[];adapters=[];qualification=[]
 for dataset in ['EMPS','F16']:
  raw=[]
  for r in manifest['records']:
   if r['dataset']!=dataset:continue
   assert sha(r['array_path'])==r['array_sha256'];a=np.load(r['array_path'],allow_pickle=False);n=len(a['u']);source=int(.4*n);cal_end=int(.8*n);fit_end=source+int(.8*(cal_end-source));raw.append((r,a['u'].copy(),a['y'].copy(),source,cal_end,fit_end))
  values=np.concatenate([np.column_stack([u[:s],y[:s]]) for _,u,y,s,_,_ in raw]);mu=values.mean(0);sd=values.std(0);assert np.all(sd>0);scaling[dataset]=dict(mean=mu.tolist(),std=sd.tolist(),source_points=len(values));rr=[]
  for r,u,y,s,c,f in raw:
   u=(u-mu[0])/sd[0];y=(y-mu[1])/sd[1];rr.append(dict(case=r['case'],u=u,y=y,source_end=s,calibration_fit_end=f,calibration_end=c,dev_origins=r['dev_origins'],validation_origins=list(range(f+128,c-128+1,256))))
   splitrows.append(dict(dataset=dataset,case=r['case'],source=[0,s],calibration_fit=[s,f],calibration_validation=[f,c],dev_origins=r['dev_origins'],internal_origins=rr[-1]['validation_origins'],source_array_sha256=r['array_sha256']))
  prepared[dataset]=rr;coef,info=learn.fit_arx([(r['u'][:r['source_end']],r['y'][:r['source_end']]) for r in rr]);p=art/(dataset+'_source_arx64.npy');np.save(p,coef,allow_pickle=False);models[dataset]=coef;modelmeta[dataset]=dict(**info,path=str(p),sha256=sha(p));print(json.dumps(dict(dataset=dataset,source_fit=info)),flush=True)
  # Explicit separate feature indexing and batch-vs-scalar recurrence qualification on source only.
  r=rr[0];t=np.array([64,65,r['source_end']-1]);d=learn.design(r['u'],r['y'],t)
  explicit=np.array([np.r_[1.,r['u'][v],r['u'][v-np.arange(1,65)],r['y'][v-np.arange(1,65)]] for v in t]);assert np.array_equal(d,explicit)
  o=256;u=r['u'][o-128:o];y=r['y'][o-128:o];fu=r['u'][o:o+128];p1=learn.rollout(coef,u[None],y[None],fu[None])[0];us=list(u);ys=list(y);p2=[]
  for v in fu:
   new=float(np.r_[1.,v,np.asarray(us[-64:][::-1]),np.asarray(ys[-64:][::-1])]@coef);p2.append(new);us.append(float(v));ys.append(new)
  difference=float(abs(p1-p2).max());assert difference<1e-10;qualification.append(dict(dataset=dataset,design_exact=True,batch_vs_scalar_recurrence_max_abs=difference))
 dump(OUT/'DATA_SPLITS75_V1.json',dict(utc=utc(),source_manifest_sha256=sha(PROJECT/'cycle74'/'outputs'/'TRAIN_DATA74_V2.json'),protocol_sha256=sha(ROOT/'PROTOCOL75_V1.md'),splits=splitrows,scaling=scaling,source_models=modelmeta,qualification=qualification,source_calibration_overlap_points=0,official_test_values_parsed=False))
 for dataset in ['EMPS','F16']:
  coef=models[dataset]
  for condition in ['identity','affine','nonlinear']:
   state('PROGRESS75_V1.json',dict(status='CONDITION_RUNNING',dataset=dataset,condition=condition,closed_rows=len(rows)))
   records=[dict(r,x=sensor(r['y'],condition)) for r in prepared[dataset]]
   # Only observed u/x calibration-fit segments cross a learner boundary.
   fitting=[(r['u'][r['source_end']:r['calibration_fit_end']].copy(),r['x'][r['source_end']:r['calibration_fit_end']].copy()) for r in records]
   validation,vids,vo=windows(records,'validation_origins');dev,ids,origins=windows(records,'dev_origins');uh,xh,fu,target=dev;scale=float(np.concatenate([x for _,x in fitting]).std());assert scale>0
   def close(method,p,info):
    path=art/(dataset+'_'+condition+'_'+method+'_dev.npz');np.savez_compressed(path,pred=p,target=target,record_ids=ids,origins=origins)
    row=dict(dataset=dataset,condition=condition,method=method,normalizer_calibration_x_std=scale,**metrics(p,target,ids,scale),model=info,arrays_path=str(path),arrays_sha256=sha(path));rows.append(row);state('PROGRESS75_V1.json',dict(status='METHOD_CLOSED',dataset=dataset,condition=condition,method=method,nrmse=row['nrmse'],closed_rows=len(rows)));print(json.dumps({k:row[k] for k in ['dataset','condition','method','nrmse','failed_records']}),flush=True)
   close('persistence',np.repeat(xh[:,-1:],128,axis=1),{})
   close('source_unadapted',learn.rollout(coef,uh,xh,fu),dict(source=modelmeta[dataset]))
   close('privileged_known_sensor',sensor(learn.rollout(coef,uh,privileged_inverse(xh,condition),fu),condition),dict(privileged=True,source=modelmeta[dataset]))
   for kind in ['affine','monotone']:
    candidates=learn.fit_maps(coef,fitting,kind)
    for m in candidates:
     p=learn.adapter_rollout(coef,m,*validation[:3]);score=metrics(p,validation[3],vids,scale);m['internal_validation_nrmse']=score['nrmse'];m['internal_failed_records']=score['failed_records']
     k=np.asarray(m['knots']);probe=np.linspace(k[0]-(k[-1]-k[0]),k[-1]+(k[-1]-k[0]),1024);roundtrip=float(abs(learn.forward_map(learn.inv_map(probe,m),m)-probe).max());m['roundtrip_max_abs']=roundtrip;assert roundtrip<1e-8
    available=[m for m in candidates if m['internal_validation_nrmse'] is not None];assert available,('all_adapters_failed',dataset,condition,kind)
    best=min(available,key=lambda m:m['internal_validation_nrmse']);ap=art/(dataset+'_'+condition+'_'+kind+'_inverse.json');dump(ap,best)
    adapters.append(dict(dataset=dataset,condition=condition,kind=kind,candidates=candidates,selected_lambda=best['lambda_identity_slopes'],path=str(ap),sha256=sha(ap)))
    k=best['knots'];close('source_'+kind+'_inverse',learn.adapter_rollout(coef,best,uh,xh,fu),dict(adapter=best,path=str(ap),sha256=sha(ap),history_extrapolation_fraction=float(np.mean((xh<k[0])|(xh>k[-1]))),source=modelmeta[dataset]))
   refit,refit_info=learn.fit_arx(fitting);path=art/(dataset+'_'+condition+'_refit_arx64.npy');np.save(path,refit,allow_pickle=False);close('calibration_arx64',learn.rollout(refit,uh,xh,fu),dict(**refit_info,path=str(path),sha256=sha(path)))
   import torch
   for seed in [2026751,2026752,2026753]:
    def save_nn(obj):
     p=art/(dataset+'_'+condition+'_narx64_seed'+str(seed)+'.pt');torch.save(obj,p);return dict(path=str(p),sha256=sha(p))
    def report(obj):
     state('NEURAL_PROGRESS75_V1.json',dict(dataset=dataset,condition=condition,**obj));print(json.dumps(dict(dataset=dataset,condition=condition,**obj)),flush=True)
    try:
     model,info=learn.neural_refit(fitting,validation,seed,save_nn,report);p=learn.neural_rollout(model,info,uh,xh,fu);close('calibration_narx64_seed'+str(seed),p,info)
    except Exception as e:
     failure=dict(error=type(e).__name__,message=str(e),traceback=traceback.format_exc(),seed=seed);dump(OUT/(dataset+'_'+condition+'_neural_failure_'+str(seed)+'.json'),failure);close('calibration_narx64_seed'+str(seed),np.full_like(target,np.nan),failure)
 result=dict(utc=utc(),status='CONSTRUCTED_OBSERVATION_CALIBRATION_DEV_CLOSED',protocol_sha256=sha(ROOT/'PROTOCOL75_V1.md'),data_splits_sha256=sha(OUT/'DATA_SPLITS75_V1.json'),rows=rows,adapter_selection=adapters,process_affinity=AFFINITY,source_ridge_fits=2,calibration_ridge_fits=6,neural_fits_attempted=18,neural_updates_per_fit=2500,official_test_numeric_values_parsed=False,constructed_sensor_shift=True,changing_physical_dynamics=False,paired_calibration_labels_available_to_learner=False,new_method=False,accepted_final_directions=0,overall_goal='ACTIVE_UNACHIEVED',local_result_files_written=0,wall_seconds=time.perf_counter()-started)
 dump(OUT/'CALIBRATION_RESULTS75_V1.json',result);state('PROGRESS75_V1.json',dict(status=result['status'],closed_rows=len(rows),wall_seconds=result['wall_seconds']));print(json.dumps(dict(status=result['status'],rows=len(rows),wall_seconds=result['wall_seconds'])),flush=True)
if __name__=='__main__':main()
