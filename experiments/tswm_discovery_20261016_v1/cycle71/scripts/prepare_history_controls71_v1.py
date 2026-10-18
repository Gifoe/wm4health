"""Fix classical controls on TRAIN only, before free-simulation VAL scores."""
import sys,pathlib,time
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common71_v1 import *
from history_controls71_v1 import *

def main():
 start=time.perf_counter();acq=read(OUT/'ACQUISITION71_V1.json');root=pathlib.Path(acq['data_root']);path=root/'state_springs5_train.npy'
 # Verify the independently confirmed canonical subset, not a same-shaped file.
 x=np.load(path,mmap_mode='r',allow_pickle=False);assert x.shape==(2048,49,5,4) and x.dtype==np.float64
 assert sha(path)=='87ffbd8fc604c9087fc1ade73bbd659d403a569872235fa4588994f754634c53'
 qualification=qualify();flat=np.asarray(x).reshape(2048,49,20);mu=flat.mean(axis=(0,1));scale=np.maximum(flat.std(axis=(0,1)),1e-8);rows=[];losses=[];selected=[]
 for lag in (1,2):
  for alpha in ALPHAS:
   errors=[];condition=[];diverged=[]
   for i,history in enumerate(flat):
    try:
     model=fit(history[:24],mu,scale,lag,alpha);pred=predict(model,25)
     with np.errstate(over='raise',invalid='raise'):err=float(np.mean((pred-history[24:])**2))
     if not np.isfinite(err):raise FloatingPointError('Nonfinite trajectory loss')
     errors.append(err);condition.append(model['condition_number'])
    except (FloatingPointError,AssertionError,np.linalg.LinAlgError) as exc:
     errors.append(np.nan);diverged.append(dict(row=i,error=type(exc).__name__,message=str(exc)))
   score=float(np.mean(errors)) if not diverged else None
   record=dict(lag=lag,alpha=alpha,mse=score,divergent_rows=diverged,all_rows_counted=True,max_condition_number=max(condition) if condition else None)
   rows.append(record);losses.append(np.asarray(errors));print(json.dumps({k:record[k] for k in ['lag','alpha','mse']})+' divergent='+str(len(diverged)),flush=True)
  candidates=[r for r in rows if r['lag']==lag and r['mse'] is not None];assert candidates,'All ordinary controls diverged; preserve evidence'
  selected.append(min(candidates,key=lambda r:r['mse']))
 artifact=storage('artifact')/'HISTORY_CONTROL_TRAIN_CALIBRATION71_V1.npz';assert not artifact.exists();np.savez_compressed(artifact,per_trajectory_mse=np.asarray(losses),mean=mu,scale=scale)
 obj=dict(utc=utc(),status='TRAIN_ONLY_HISTORY_CONTROLS_FIXED',source_data_sha256=sha(path),protocol_sha256=sha(ROOT/'FREE_SIMULATION_PROTOCOL71_V1.md'),implementation_sha256=sha(ROOT/'scripts'/'history_controls71_v1.py'),qualification=qualification,train_rows=2048,context_steps=24,forecast_steps=25,val_rows_accessed=0,eval_rows_accessed=0,alphas=list(ALPHAS),grid_results=rows,selected=selected,coordinate_mean=mu.tolist(),coordinate_scale=scale.tolist(),raw_artifact_path=str(artifact),raw_artifact_sha256=sha(artifact),wall_seconds=time.perf_counter()-start,new_method=False,accepted_final_directions=0)
 dump(OUT/'HISTORY_CONTROL_TRAIN_CALIBRATION71_V1.json',obj);print(json.dumps(dict(status=obj['status'],selected=selected,wall_seconds=obj['wall_seconds'])),flush=True)

if __name__=='__main__':main()
