"""Ordinary TRAIN-fit, DEV-only recursive controls; no official test targets."""
import sys,pathlib,time
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common74_v1 import *
sys.path.insert(0,str(PROJECT/'cycle71'/'scripts'))
from runtime_affinity71_v1 import configure
AFFINITY=configure('efficient')
import numpy as np
def design(u,y,t,lag):
 return np.column_stack([np.ones(len(t)),u[t]]+[u[t-j] for j in range(1,lag+1)]+[y[t-j] for j in range(1,lag+1)])
def feature(uh,yh,current,lag):return np.r_[1.,current,np.asarray(uh[-lag:][::-1]),np.asarray(yh[-lag:][::-1])]
def rollout(coef,uh,yh,future_u,lag):
 # This API has no future-output argument; history is copied before prediction.
 us=list(np.array(uh,dtype=np.float64,copy=True));ys=list(np.array(yh,dtype=np.float64,copy=True));out=[]
 with np.errstate(over='ignore',invalid='ignore'):
  for v in future_u:
   value=float(feature(us,ys,float(v),lag)@coef);out.append(value);us.append(float(v));ys.append(value)
 return np.asarray(out)
def main():
 started=time.perf_counter();m=read(OUT/'TRAIN_DATA74_V1.json');assert not m['official_test_numeric_values_parsed'];raw={};rows=[];art=storage('artifact')/'canonical_arx_v1';art.mkdir(exist_ok=False);qualified=[]
 for r in m['records']:
  assert sha(r['array_path'])==r['array_sha256'];a=np.load(r['array_path'],allow_pickle=False);sc=m['scaling'][r['dataset']];u=(a['u']-sc['mean'][0])/sc['std'][0];y=(a['y']-sc['mean'][1])/sc['std'][1];raw[r['case']]=(u,y)
 for name in ['EMPS','F16']:
  records=[r for r in m['records'] if r['dataset']==name];models=[]
  for lag in [4,16,64]:
   dim=2+2*lag;gram=np.zeros((dim,dim));rhs=np.zeros(dim);nfit=0
   for r in records:
    u,y=raw[r['case']];cut=r['train_end_exclusive'];times=np.array([lag,lag+1,cut-2,cut-1]);x=design(u,y,times,lag)
    for i,t in enumerate(times):assert np.array_equal(x[i],feature(u[t-lag:t],y[t-lag:t],u[t],lag))
    for start in range(lag,cut,8192):
     ix=np.arange(start,min(start+8192,cut));x=design(u,y,ix,lag);gram+=x.T@x;rhs+=x.T@y[ix];nfit+=len(ix)
   ridge=.001*float(np.diag(gram)[1:].mean());penalty=np.eye(dim)*ridge;penalty[0,0]=0;coef=np.linalg.solve(gram+penalty,rhs);assert np.isfinite(coef).all();normal_residual=float(abs((gram+penalty)@coef-rhs).max());cp=art/(name+'_lag'+str(lag)+'_coef.npy');np.save(cp,coef,allow_pickle=False)
   models.append(dict(method='ridge_arx_lag'+str(lag),lag=lag,coef=coef,coefficient_path=str(cp),coefficient_sha256=sha(cp),training_rows=nfit,regularization=ridge,normal_equation_max_abs_residual=normal_residual))
  for model in models+[dict(method='persistence'),dict(method='train_mean')]:
   per_record=[];all_errors=[];nonfinite=0;npz={}
   for r in records:
    u,y=raw[r['case']];pred=[];target=[]
    for o in r['dev_origins']:
     if model['method'].startswith('ridge'):
      lag=model['lag'];p=rollout(model['coef'],u[o-128:o],y[o-128:o],u[o:o+128],lag)
      if not qualified:
       second=rollout(model['coef'],u[o-128:o],y[o-128:o],u[o:o+128].copy(),lag);assert np.array_equal(p,second)
       qualified.append(dict(case=r['case'],lag=lag,vectorized_design_matches_explicit_indexing=True,rollout_future_output_argument=False,history_copied=True))
     elif model['method']=='persistence':p=np.full(128,y[o-1])
     else:p=np.zeros(128)
     pred.append(p);target.append(y[o:o+128].copy())
    pred=np.asarray(pred);target=np.asarray(target);finite=bool(np.isfinite(pred).all());nonfinite+=int(not finite);error=pred-target
    item=dict(case=r['case'],origins=len(pred),forecast_points=pred.size,all_predictions_finite=finite,normalized_rmse=float(np.sqrt(np.mean(error**2))) if finite else None,normalized_mae=float(np.mean(abs(error))) if finite else None,raw_rmse=float(np.sqrt(np.mean(error**2))*m['scaling'][name]['std'][1]) if finite else None)
    per_record.append(item);npz[r['case']+'_pred']=pred;npz[r['case']+'_target']=target;npz[r['case']+'_origins']=np.asarray(r['dev_origins'])
    if finite:all_errors.append(error.ravel())
   dest=art/(name+'_'+model['method']+'_dev_predictions.npz');np.savez_compressed(dest,**npz);errs=np.concatenate(all_errors) if all_errors else np.array([])
   row=dict(dataset=name,method=model['method'],per_record=per_record,failed_records=nonfinite,all_records_included_in_aggregate=nonfinite==0,normalized_rmse=float(np.sqrt(np.mean(errs**2))) if nonfinite==0 else None,normalized_mae=float(np.mean(abs(errs))) if nonfinite==0 else None,array_path=str(dest),array_sha256=sha(dest),model={k:v for k,v in model.items() if k!='coef'},train_only_scaling=m['scaling'][name]);rows.append(row);print(json.dumps({k:row[k] for k in ['dataset','method','normalized_rmse','normalized_mae','failed_records']}),flush=True)
 selected={}
 for name in ['EMPS','F16']:
  good=[r for r in rows if r['dataset']==name and r['method'].startswith('ridge') and r['failed_records']==0];selected[name]=min(good,key=lambda r:r['normalized_rmse'])['method'] if good else None
 result=dict(utc=utc(),status='SIX_ORDINARY_ARX_TRAIN_FITS_AND_DEV_FREE_ROLLOUTS_CLOSED',data_manifest_sha256=sha(OUT/'TRAIN_DATA74_V1.json'),protocol_sha256=sha(ROOT/'DATA_PROTOCOL74_V1.md'),scaling_clarification_sha256=sha(ROOT/'DATA_SCALING_CLARIFICATION74_V1.md'),rows=rows,selected_control_lag_from_dev=selected,qualification=qualified,process_affinity=AFFINITY,new_ridge_fits=6,neural_fits=0,official_test_output_values_parsed=False,constructed_sensor_shift_scores=0,new_method=False,accepted_final_directions=0,overall_goal='ACTIVE_UNACHIEVED',local_result_files_written=0,wall_seconds=time.perf_counter()-started)
 dump(OUT/'CANONICAL_ARX74_V1.json',result);print(json.dumps(dict(status=result['status'],selected=selected,wall_seconds=result['wall_seconds'])),flush=True)
if __name__=='__main__':main()
