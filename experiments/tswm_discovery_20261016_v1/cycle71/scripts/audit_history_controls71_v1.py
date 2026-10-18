"""Selected-control numerical audit prompted by high TRAIN condition numbers."""
import sys,pathlib
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common71_v1 import *
from history_controls71_v1 import *

cal=read(OUT/'HISTORY_CONTROL_TRAIN_CALIBRATION71_V1.json');acq=read(OUT/'ACQUISITION71_V1.json');p=pathlib.Path(acq['data_root'])/'state_springs5_train.npy';assert sha(p)==cal['source_data_sha256'];mu=np.array(cal['coordinate_mean']);scale=np.array(cal['coordinate_scale']);data=np.load(p,mmap_mode='r',allow_pickle=False).reshape(2048,49,20);rows=[]
for chosen in cal['selected']:
 lag=chosen['lag'];alpha=chosen['alpha'];models=[fit(x[:24],mu,scale,lag,alpha) for x in data];idx=int(np.argmax([m['condition_number'] for m in models]));model=models[idx];z=(data[idx,:24]-mu)/scale;X=[];Y=[]
 for t in range(lag-1,23):X.append(np.concatenate([z[t-j] for j in range(lag)]+[np.ones(1)]));Y.append(z[t+1]-z[t])
 X=np.asarray(X);Y=np.asarray(Y);root=np.sqrt(len(X)*alpha)*np.eye(X.shape[1]);root[-1,-1]=0
 reference=np.linalg.lstsq(np.vstack([X,root]),np.vstack([Y,np.zeros((X.shape[1],20))]),rcond=None)[0];ref=dict(model);ref['coef']=reference;forecast=predict(model,25);check=predict(ref,25)
 coefficient_error=float(abs(reference-model['coef']).max());forecast_error=float(abs(forecast-check).max());scale_error=forecast_error/max(1.,float(abs(check).max()));mse_a=float(np.mean((forecast-data[idx,24:])**2));mse_b=float(np.mean((check-data[idx,24:])**2));relative_mse_error=abs(mse_a-mse_b)/max(mse_b,1e-15)
 rows.append(dict(lag=lag,alpha=alpha,worst_train_row=idx,condition_number=model['condition_number'],coefficient_max_abs=coefficient_error,forecast_max_abs=forecast_error,forecast_scaled_error=scale_error,normal_equation_mse=mse_a,augmented_lstsq_mse=mse_b,relative_mse_difference=relative_mse_error))
obj=dict(utc=utc(),status='NUMERICAL_COMPARISON_RECORDED_NOT_A_NEW_METHOD',calibration_sha256=sha(OUT/'HISTORY_CONTROL_TRAIN_CALIBRATION71_V1.json'),checks=rows,train_rows=2048,val_rows_accessed=0,eval_rows_accessed=0,accepted_final_directions=0)
dump(OUT/'HISTORY_CONTROL_NUMERICAL_AUDIT71_V1.json',obj);print(json.dumps(obj),flush=True)
