"""Same-information source-mode and standard corrected EM;validated batching,no novel method."""
import native_em_positive_control as common
import batched_em as batch
import csv,json,pathlib,time
import numpy as np
import actuated_Koopman_EM_4 as native
ROOT=common.ROOT;SOURCE=common.OUT;OUT=ROOT/'outputs/controlled_em_study'

def save(p,iteration,ll,path):
 for name,value in p.items():assert np.isfinite(value).all(),name
 for name in ['Sig_v','Sig_w','Sig_0']:assert np.linalg.eigvalsh(p[name]).min()>-1e-10,(name,iteration)
 np.savez_compressed(path,**p,iteration=np.array(iteration),likelihood=np.array(ll))
 return dict(path=path.relative_to(OUT).as_posix(),sha256=common.digest(path),iteration=iteration,likelihood=float(ll))

def fit(algorithm,seed,y,u):
 folder=OUT/algorithm/f'seed{seed}';folder.mkdir(parents=True);started=time.monotonic();scaled=.01*u;trace=[];checkpoints=[];resumed=0
 p=batch.initial(seed,u,common.reflect_eigs);m,s,c,ll=batch.estep(p,y,scaled,implicit=True)
 checkpoints.append(save(p,0,ll,folder/'weights_init.npz'));best=None;best_l=-np.inf;previous=None;delta=np.inf
 if algorithm=='source_mode' and seed==20262711:
  original=SOURCE/f'seed{seed}'
  paths=sorted(original.glob('weights_iter*.npz'),key=lambda f:int(f.stem.split('iter')[1]))
  for path in paths:
   data=np.load(path);iteration=int(data['iteration']);assert np.isfinite(data['likelihood'])
   copied=folder/path.name;copied.write_bytes(path.read_bytes())
   cp=dict(path=copied.relative_to(OUT).as_posix(),sha256=common.digest(copied),iteration=iteration,likelihood=float(data['likelihood']))
   checkpoints.append(cp)
   if iteration>0 and cp['likelihood']>best_l:best=cp;best_l=cp['likelihood']
  latest=max(checkpoints,key=lambda cp:cp['iteration']);data=np.load(OUT/latest['path']);resumed=latest['iteration']
  p={name:data[name].copy() for name in common.PARAMETERS};m,s,c,ll=batch.estep(p,y,scaled)
  assert abs(ll-latest['likelihood'])<1e-4
  original_trace=json.loads((original/'trace.json').read_text());trace=[dict(row,backend='unmodified_native') for row in original_trace if row['iteration']<=resumed]
  previous=(resumed,float(ll));common.progress(phase='resume',algorithm=algorithm,seed=seed,iteration=resumed,source_sha256=latest['sha256'])
 for iteration in range(resumed+1,1001):
  p=batch.mstep(p,y,scaled,m,s,c,corrected=algorithm=='corrected_em');m,s,c,ll=batch.estep(p,y,scaled)
  if iteration==1 or iteration%10==0:
   if previous is not None:delta=(ll-previous[1])/(iteration-previous[0])
   previous=(iteration,ll)
   row=dict(seed=seed,iteration=iteration,likelihood=float(ll),average_ll_gain=float(delta) if np.isfinite(delta) else None,elapsed_batched_s=time.monotonic()-started,backend='batched_numpy')
   trace.append(row);cp=save(p,iteration,ll,folder/f'weights_iter{iteration}.npz');checkpoints.append(cp)
   if ll>best_l:best=cp;best_l=ll
   common.write(folder/'trace.json',trace)
   if iteration%100==0:common.progress(phase='fit_amended',algorithm=algorithm,**row)
   if iteration>=100 and delta<=.01:break
 stopping='train_likelihood_tolerance' if delta<=.01 else 'iteration_cap_underconverged'
 result=dict(algorithm=algorithm,seed=seed,total_iterations=iteration,stopping=stopping,last_average_ll_gain=float(delta),
   best=best,checkpoint100=next(cp for cp in checkpoints if cp['iteration']==100),checkpoints=checkpoints,
   runtime_batched_s=time.monotonic()-started,resumed_native_iteration=resumed)
 common.write(folder/'fit.json',result);return result

def forecast(p,mean,cov,u):
 full=batch.evolution(p,.01*u);means=[];variances=[]
 for t in range(u.shape[1]):
  a=full[:,t,1:,1:];b=full[:,t,1:,0]
  mean=(a@mean[:,:,None])[:,:,0]+b;cov=p['Sig_v']+a@cov@a.transpose(0,2,1)
  means.append(mean@p['C'][0]+p['h'][0]);variances.append((cov@p['C'][0])@p['C'][0]+p['Sig_w'][0,0])
 return np.stack(means,axis=1),np.stack(variances,axis=1)

def main():
 assert not OUT.exists();OUT.mkdir();common.OUT=OUT;started=time.monotonic()
 proof=json.loads((ROOT/'outputs/backend_validation/results.json').read_text());assert proof['status']=='passed'
 oldstatus=json.loads((ROOT/'outputs/jobs/native_em_positive_v1/status.json').read_text());assert oldstatus['status']=='failed'
 stop_receipt=ROOT/'outputs/controlled_native_stop.json';assert stop_receipt.exists()
 receipt=json.loads((SOURCE/'fit_input_receipt.json').read_text());datasets={}
 for row in receipt['data']:
  path=SOURCE/row['path'];assert common.digest(path)==row['sha256'];datasets[row['name']]=dict(np.load(path))
 train=datasets['train_continuation'];y=np.ascontiguousarray(train['y'][:,:250]);u=np.ascontiguousarray(train['u'][:,:249]);fits=[]
 assert common.array_digest(y)==receipt['y_sha256'] and common.array_digest(u)==receipt['u_unscaled_sha256']
 for algorithm in ['source_mode','corrected_em']:
  for seed in common.FIT_SEEDS:fits.append(fit(algorithm,seed,y,u))
 common.write(OUT/'all_fits_completed_before_evaluation.json',dict(fits=fits,elapsed_s=time.monotonic()-started,protocol_sha256=common.digest(ROOT/'PROTOCOL.md'),amendment_sha256=common.digest(ROOT/'AMENDMENT_SOURCE_MATH.md')))
 arx=common.fit_arx(train);records=[];predictions=[];data_gates=[];native_forecast_error=0.
 for name,data in datasets.items():
  exact=common.exact_physics(data['x'][:,0],data['u'][:,:,1]);actual=data['x'][:,1:,1]
  assert np.allclose(exact,actual,atol=1e-4,rtol=5e-6)
  data_gates.append(dict(dataset=name,physical_max_abs_error=float(np.max(np.abs(exact-actual)))))
  for prefix in ([250] if name=='train_continuation' else common.PREFIXES):
   base=[('persistence',np.repeat(data['y'][:,prefix-1,0,None],common.HORIZON,axis=1),None),('arx',common.arx_predict(arx,data,prefix),None)]
   oracle=common.exact_physics(data['x'][:,prefix-1],data['u'][:,prefix-1:prefix-1+common.HORIZON,1]);base.append(('oracle_true_state_physics',oracle,np.full_like(oracle,.01)))
   for method,mean,var in base:
    path=OUT/f'prediction_{name}_p{prefix}_{method}.npz';parts=dict(mean=mean,prefix=np.array(prefix))
    if var is not None:parts['variance']=var
    np.savez_compressed(path,**parts);predictions.append(dict(path=path.name,sha256=common.digest(path)))
    records.append(dict(dataset=name,prefix=prefix,algorithm='baseline',method=method,seed=-1,checkpoint='none',**common.scores(mean,var,data,prefix)))
 for fit_result in fits:
  algorithm=fit_result['algorithm'];seed=fit_result['seed']
  for label,weight in [('iteration100',fit_result['checkpoint100']),('trainbest_final',fit_result['best'])]:
   values=np.load(OUT/weight['path']);p={name:values[name] for name in common.PARAMETERS}
   for name,data in datasets.items():
    for prefix in ([250] if name=='train_continuation' else common.PREFIXES):
     m,cov,_,_=batch.estep(p,np.ascontiguousarray(data['y'][:,:prefix]),.01*np.ascontiguousarray(data['u'][:,:prefix-1]))
     for convention,offset_index in [('filtered_prefix',-1),('native_smoothed_initial',0)]:
      start=prefix-1 if offset_index==-1 else 0;plan=data['u'][:,start:prefix-1+common.HORIZON]
      initial=m[:,offset_index].copy();initial_cov=cov[:,offset_index].copy()
      mean_all,var_all=forecast(p,initial.copy(),initial_cov.copy(),plan)
      native_mean,native_cov,_,_=native._predict_dynamics.py_func(initial[:2].copy(),initial_cov[:2].copy(),batch.evolution(p,.01*plan[:2]),p['Sig_v'],p['C'],p['h'],p['Sig_w'])
      native_forecast_error=max(native_forecast_error,float(np.max(np.abs(native_mean[:,:,0]-mean_all[:2]))),float(np.max(np.abs(native_cov[:,:,0,0]-var_all[:2]))))
      assert np.allclose(native_mean[:,:,0],mean_all[:2],atol=2e-7,rtol=1e-7) and np.allclose(native_cov[:,:,0,0],var_all[:2],atol=2e-7,rtol=1e-7)
      offset=prefix-1-start;mean=mean_all[:,offset:];variance=var_all[:,offset:];assert mean.shape==(len(data['y']),common.HORIZON)
      path=OUT/f'prediction_{name}_p{prefix}_{algorithm}_seed{seed}_{label}_{convention}.npz'
      np.savez_compressed(path,mean=mean,variance=variance,latent_initial_mean=initial,latent_initial_covariance=initial_cov,prefix=np.array(prefix),start_time=np.array(start),weights_relative=np.array(weight['path']))
      predictions.append(dict(path=path.name,sha256=common.digest(path)))
      records.append(dict(dataset=name,prefix=prefix,algorithm=algorithm,method=convention,seed=seed,checkpoint=label,weight_iteration=weight['iteration'],**common.scores(mean,variance,data,prefix)))
   common.progress(phase='evaluate_amended',algorithm=algorithm,seed=seed,checkpoint=label,rows=len(records))
 fields=list(dict.fromkeys(k for row in records for k in row))
 with (OUT/'rows.csv').open('w',newline='',encoding='utf-8') as f:
  writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(records)
 common.write(OUT/'data_gate.json',data_gates)
 result=dict(status='completed',n_rows=len(records),fits=fits,predictions=predictions,data=receipt['data'],data_root='../native_em',
   native_forecast_max_abs_error=native_forecast_error,runtime_s=time.monotonic()-started,
   scope='Existing controlled EM,source-equivalent and standard covariance-transpose correction;not new framework',
   native_stop_receipt_sha256=common.digest(stop_receipt),backend_validation_sha256=common.digest(ROOT/'outputs/backend_validation/results.json'),
   all_fits_receipt_sha256=common.digest(OUT/'all_fits_completed_before_evaluation.json'),
   protocol_sha256=common.digest(ROOT/'PROTOCOL.md'),amendment_sha256=common.digest(ROOT/'AMENDMENT_SOURCE_MATH.md'),script_sha256=common.digest(pathlib.Path(__file__)),accepted_final_directions=0)
 common.write(OUT/'results.json',result);common.progress(phase='amended_complete',rows=len(records),runtime_s=result['runtime_s'])

if __name__=='__main__':main()
