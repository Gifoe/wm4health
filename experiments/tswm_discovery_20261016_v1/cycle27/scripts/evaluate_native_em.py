"""Evaluation-only wrapper repair. Immutable native fits/data; no retraining or test selection."""
import native_em_positive_control as common
import csv,json,pathlib,time
import numpy as np

ROOT=common.ROOT;SOURCE=common.OUT;OUT=ROOT/'outputs/native_em_evaluation_v2'
def main():
 assert not OUT.exists(),'Refuse overwriting an evaluation'
 receipt=SOURCE/'fits_completed_before_evaluation.json';completed=json.loads(receipt.read_text())
 status=json.loads((ROOT/'outputs/jobs/native_em_positive_v1/status.json').read_text())
 assert status['status']=='failed','Observe exact original terminal status before evaluation-only recovery'
 assert not (SOURCE/'fit_failure_summary.json').exists(),'Training failed;do not silently score incomplete fits'
 fits=completed['fits'];assert [r['seed'] for r in fits]==common.FIT_SEEDS
 fit_input=json.loads((SOURCE/'fit_input_receipt.json').read_text());data_files=fit_input['data']
 for fit in fits:
  for row in fit['checkpoints']:assert common.digest(SOURCE/f"seed{fit['seed']}"/row['path'])==row['sha256']
 datasets={}
 for row in data_files:
  path=SOURCE/row['path'];assert common.digest(path)==row['sha256'];datasets[row['name']]=dict(np.load(path))
 OUT.mkdir(parents=True);common.OUT=OUT;started=time.monotonic()
 train=datasets['train_continuation'];arx=common.fit_arx(train);records=[];predictions=[];data_gates=[]
 for name,data in datasets.items():
  exact=common.exact_physics(data['x'][:,0],data['u'][:,:,1]);actual=data['x'][:,1:,1];difference=np.abs(exact-actual)
  assert np.allclose(exact,actual,atol=1e-4,rtol=5e-6)
  data_gates.append(dict(dataset=name,max_physical_output_abs_error=float(difference.max()),rms_physical_output_error=float(np.sqrt(np.mean(difference**2)))))
  prefixes=[250] if name=='train_continuation' else common.PREFIXES
  for prefix in prefixes:
   base=[('persistence',np.repeat(data['y'][:,prefix-1,0,None],common.HORIZON,axis=1),None),('arx',common.arx_predict(arx,data,prefix),None)]
   oracle=common.exact_physics(data['x'][:,prefix-1],data['u'][:,prefix-1:prefix-1+common.HORIZON,1]);base.append(('oracle_true_state_physics',oracle,np.full_like(oracle,.01)))
   for method,mean,var in base:
    path=OUT/f'prediction_{name}_p{prefix}_{method}.npz';parts=dict(mean=mean,prefix=np.array(prefix))
    if var is not None:parts['variance']=var
    np.savez_compressed(path,**parts);predictions.append(dict(path=path.name,sha256=common.digest(path)))
    records.append(dict(dataset=name,prefix=prefix,method=method,seed=-1,checkpoint='none',**common.scores(mean,var,data,prefix)))
 for fit in fits:
  seed=fit['seed']
  for checkpoint,weight in [('iteration100',fit['checkpoint100']),('trainbest_final',fit['best'])]:
   weight_path=SOURCE/f'seed{seed}'/weight['path'];model=common.load_model(weight_path)
   for name,data in datasets.items():
    prefixes=[250] if name=='train_continuation' else common.PREFIXES
    for prefix in prefixes:
     history=np.ascontiguousarray(data['y'][:,:prefix]);past_inputs=np.ascontiguousarray(common.DT*data['u'][:,:prefix-1])
     means,covs=model.infer_latent_states(history,past_inputs,explicit_time_step=True)
     for convention,start_time,index in [('filtered_prefix',prefix-1,-1),('native_smoothed_initial',0,0)]:
      plan=np.ascontiguousarray(common.DT*data['u'][:,start_time:prefix-1+common.HORIZON])
      pred,predcov,_,_=model.predict_dynamics(means[:,index].copy(),covs[:,index].copy(),plan,explicit_time_step=True)
      offset=prefix-1-start_time;mean=pred[:,offset:,0];variance=predcov[:,offset:,0,0];assert mean.shape==(len(history),common.HORIZON)
      path=OUT/f'prediction_{name}_p{prefix}_seed{seed}_{checkpoint}_{convention}.npz'
      np.savez_compressed(path,mean=mean,variance=variance,latent_initial_mean=means[:,index],latent_initial_covariance=covs[:,index],prefix=np.array(prefix),start_time=np.array(start_time),weights_relative=np.array('../native_em/'+f'seed{seed}/'+weight['path']))
      predictions.append(dict(path=path.name,sha256=common.digest(path)))
      records.append(dict(dataset=name,prefix=prefix,method=convention,seed=seed,checkpoint=checkpoint,weight_iteration=weight['iteration'],**common.scores(mean,variance,data,prefix)))
    common.progress(phase='evaluate_repaired',seed=seed,checkpoint=checkpoint,dataset=name,rows=len(records))
 common.write(OUT/'data_gate.json',data_gates)
 fieldnames=list(dict.fromkeys(k for row in records for k in row))
 with (OUT/'rows.csv').open('w',newline='',encoding='utf-8') as f:
  writer=csv.DictWriter(f,fieldnames=fieldnames);writer.writeheader();writer.writerows(records)
 result=dict(status='completed',scope='Evaluation-only wrapper return-unpacking repair;no weights,data,protocol or settings changed',
   n_rows=len(records),fit_seeds=common.FIT_SEEDS,fits=fits,data=data_files,data_root='../native_em',predictions=predictions,
   runtime_s=time.monotonic()-started,training_receipt_sha256=common.digest(receipt),original_job_status=status,
   script_sha256=common.digest(pathlib.Path(__file__)),protocol_sha256=common.digest(ROOT/'PROTOCOL.md'),accepted_final_directions=0)
 common.write(OUT/'results.json',result);common.progress(phase='evaluation_completed',rows=len(records),runtime_s=result['runtime_s'])

if __name__=='__main__':main()
