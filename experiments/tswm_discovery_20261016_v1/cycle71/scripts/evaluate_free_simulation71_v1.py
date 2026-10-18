"""Frozen development assessment; all three complete fits required, EVAL closed."""
import sys,pathlib,time,types,copy,random,gc
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common71_v1 import *
import numpy as np
import history_controls71_v1 as controls

SEEDS=(2026711,2026712,2026713)

def campaign_gate():
 assert read(OUT/'campaign71_v3_exit.json')['exit_code']==0
 campaign=read(OUT/'CAMPAIGN71_V3.json')
 assert campaign['status']=='THREE_NATIVE_SUBSET_FITS_CLOSED_TRAIN_VAL_ONLY'
 assert campaign['module_updates']==96000 and campaign['reserved_eval_rows_accessed']==0
 assert [r['seed'] for r in campaign['fits']]==list(SEEDS)
 for r in campaign['fits']:
  stem='formal_seed'+str(r['seed'])+'_v3';result=read(OUT/(stem+'_result.json'))
  assert read(OUT/(stem+'_exit.json'))['exit_code']==0 and result['epochs']==2000
  assert result['module_updates']==result['proposal_updates']==32000
  assert sha(OUT/(stem+'_result.json'))==r['result_sha256']
  assert result['selected']==r['selected'] and sha(r['selected']['path'])==r['selected']['sha256']
 return campaign

def metrics(pred,truth):
 assert pred.shape==truth.shape==(256,50,5,4)
 finite=np.isfinite(pred).all(axis=(1,2,3));bad=np.flatnonzero(~finite).tolist()
 rows=[]
 if not bad:
  try:
   with np.errstate(over='raise',invalid='raise'):
    diff=pred.astype(np.float64)-truth;square=diff**2
    for H in (1,10,25,50):
     rows.append(dict(horizon=H,prefix_mse=float(square[:,:H].mean()),prefix_mae=float(abs(diff[:,:H]).mean()),endpoint_mse=float(square[:,H-1].mean()),coordinate_mse=square[:,:H].mean(axis=(0,1,2)).tolist()))
   assert all(np.isfinite(r['prefix_mse']) for r in rows)
  except (FloatingPointError,AssertionError):return dict(status='METRIC_OVERFLOW',nonfinite_rows=bad,metrics=None)
 return dict(status='FINITE_ALL_TRAJECTORIES' if not bad else 'NONFINITE_TRAJECTORIES',nonfinite_rows=bad,total_trajectories=256,metrics=rows if not bad else None)

def save_array(name,**arrays):
 p=storage('artifact')/(name+'.npz');assert not p.exists();np.savez_compressed(p,**arrays)
 return dict(path=str(p),sha256=sha(p))

def history_datasets(native,history):
 out=[]
 for i,h in enumerate(history):
  t=native.torch.as_tensor(h,dtype=native.torch.float32,device='cuda')
  out.append(types.SimpleNamespace(TrainInput=t,TrainOutput=t[1:],ValInput=t,ValOutput=t[1:],Edges=np.zeros(20,dtype=np.int64),structure_idx=i))
 return out

def reseed(native,seed):
 random.seed(seed);np.random.seed(seed);native.torch.manual_seed(seed);native.torch.cuda.manual_seed_all(seed)

def assignments(structures):return np.asarray([s['edge_idx_inv'] for s in structures],dtype=np.int64)

def assign(native,bg,history,edges):
 structures,_=native.context_inference(bg,history,steps=0)
 for s,e in zip(structures,edges):bg.S.update_structure_to(s,e.tolist())
 return structures

def calibrate_orientation(native,bg,train,edges):
 losses=[]
 for flip in (0,1):
  structures=assign(native,bg,train[:128],edges[:128]^flip)
  with native.torch.no_grad():vals=bg.batched_evaluate(structures,history_datasets(native,train[:128]),'Train')[0]
  loss=float(np.mean([float(v.item()) for v in vals]));assert np.isfinite(loss);losses.append(loss)
 return dict(flip=int(losses[1]<losses[0]),train_rows=128,native_reset_block_history_mse=losses,val_labels_used_for_selection=False)

def neural_fit(native,fit,train,train_edges,histories,truth,val_edges):
 seed=fit['seed'];tag='free_sim_seed'+str(seed)+'_v1';start=time.perf_counter();reseed(native,seed+40000)
 ck=fit['selected'];assert sha(ck['path'])==ck['sha256'];state=native.torch.load(ck['path'],map_location='cpu',weights_only=False)
 args=native.parse_native(seed,1,2048,tag);bg=native.InstrumentedNative(native.HGNN_Structure(args),args,tag)
 bg.L.load_state_dict(state['module_state'],strict=True);bg.encoder.load_state_dict(state['proposal_state'],strict=True);bg.L.eval();bg.encoder.eval();before=native.weights_hash(bg)
 assert sum(v.numel() for v in bg.L.parameters())==467204 and sum(v.numel() for v in bg.encoder.parameters())==2735106
 orientation=calibrate_orientation(native,bg,train,train_edges);records=[]
 for arm in ('encoder_argmax','native_search100','native_search1000','privileged_true_assignments'):
  outputs=[];graphs=[];searches=[];history_loss=[];arm_start=time.perf_counter()
  for offset in range(0,256,32):
   history=histories[offset:offset+32];reseed(native,seed+50000+offset)
   if arm=='encoder_argmax':
    with native.torch.no_grad():probs=bg.encoder(native.torch.as_tensor(history,dtype=native.torch.float32,device='cuda').transpose(1,2).contiguous(),bg.rel_rec,bg.rel_send).detach().cpu().numpy()
    structures=assign(native,bg,history,np.argmax(probs,axis=-1));search=dict(steps=0)
   elif arm=='privileged_true_assignments':structures=assign(native,bg,history,val_edges[offset:offset+32]^orientation['flip']);search=dict(steps=0,privileged_graph_labels=True)
   else:structures,search=native.context_inference(bg,history,steps=100 if arm=='native_search100' else 1000)
   with native.torch.no_grad():vals=bg.batched_evaluate(structures,history_datasets(native,history),'Train')[0]
   history_loss.extend(float(v.item()) for v in vals);outputs.append(native.rollout(bg,structures,history[:,-1],50,0.));graphs.append(assignments(structures));searches.append(dict(offset=offset,**search))
  pred=np.concatenate(outputs);graph=np.concatenate(graphs);record=dict(seed=seed,arm=arm,privileged=(arm=='privileged_true_assignments'),checkpoint=ck,orientation=orientation,mean_native_history_mse=float(np.mean(history_loss)),graph_assignment_accuracy=float(np.mean(graph==(val_edges^orientation['flip']))),search=searches,wall_seconds=time.perf_counter()-arm_start,**metrics(pred,truth))
  record['arrays']=save_array(tag+'_'+arm,predictions=pred,assignments=graph,history_mse=np.asarray(history_loss));records.append(record)
  dump(OUT/(tag+'_'+arm+'_result.json'),record);print(json.dumps(dict(seed=seed,arm=arm,status=record['status'],metrics=record['metrics'],wall_seconds=record['wall_seconds'])),flush=True)
 assert native.weights_hash(bg)==before and sha(ck['path'])==ck['sha256'];assert bg.module_updates71==bg.proposal_updates71==0
 bg.writer.close();obj=dict(seed=seed,checkpoint=ck,arms=records,weights_unchanged_sha256=before,source_bodies=native.EXECUTED_BODIES,module_updates=0,proposal_updates=0,wall_seconds=time.perf_counter()-start,reserved_eval_rows_accessed=0)
 dump(OUT/(tag+'_result.json'),obj);del state,bg;gc.collect();native.torch.cuda.empty_cache();return obj

def main():
 start=time.perf_counter();campaign=campaign_gate();acq=read(OUT/'ACQUISITION71_V1.json');data=pathlib.Path(acq['data_root']);cal_path=OUT/'HISTORY_CONTROL_TRAIN_CALIBRATION71_V1.json'
 assert sha(cal_path)=='01196c84f8982b4e70002c60e793079ed79a8cf2a958a5c03fe4b61cb5729dc0'
 cal=read(cal_path);assert cal['val_rows_accessed']==cal['eval_rows_accessed']==0
 assert sha(ROOT/'FREE_SIMULATION_PROTOCOL71_V1.md')==cal['protocol_sha256'] and sha(ROOT/'scripts'/'history_controls71_v1.py')==cal['implementation_sha256']
 numerical=read(OUT/'HISTORY_CONTROL_NUMERICAL_AUDIT71_V1.json');assert numerical['calibration_sha256']==sha(cal_path)
 assert all(r['forecast_scaled_error']<1e-7 and r['relative_mse_difference']<1e-4 for r in numerical['checks'])
 assert sha(data/'state_springs5_train.npy')==cal['source_data_sha256']
 assert sha(data/'state_springs5_test.npy')=='2484c0a2f00528087cd59f392cd449173c95e513ced3efa033599ae625491e64'
 assert sha(data/'edges_springs5_train.npy')=='a99f886f4b057fb95372218d78e0902c51eab8f6a6c11d643311f53ab0805f12'
 assert sha(data/'edges_springs5_test.npy')=='9b7d18a19fa6ad8c37a9d1b30b83712c77cb440466901409bd7d619c154fcfc3'
 train=np.load(data/'state_springs5_train.npy',mmap_mode='r',allow_pickle=False);val=np.load(data/'state_springs5_test.npy',mmap_mode='r',allow_pickle=False)[:256].copy();assert val.shape==(256,99,5,4)
 histories=val[:,:49].copy();truth=val[:,49:].copy();val_edges=np.load(data/'edges_springs5_test.npy',allow_pickle=False)[:256].copy();train_edges=np.load(data/'edges_springs5_train.npy',allow_pickle=False);mu=np.asarray(cal['coordinate_mean']);scale=np.asarray(cal['coordinate_scale']);classical=[]
 for lag in (0,1,2):
  begin=time.perf_counter();pred=[];failures=[];alpha=None if lag==0 else next(r['alpha'] for r in cal['selected'] if r['lag']==lag)
  for i,h in enumerate(histories):
   try:p=np.repeat(h[-1:,:,:],50,axis=0) if lag==0 else controls.predict(controls.fit(h,mu,scale,lag,alpha),50).reshape(50,5,4)
   except (FloatingPointError,AssertionError,np.linalg.LinAlgError) as exc:p=np.full((50,5,4),np.nan);failures.append(dict(row=i,error=type(exc).__name__,message=str(exc)))
   pred.append(p)
  pred=np.asarray(pred);name='persistence' if lag==0 else 'history_VAR'+str(lag);record=dict(arm=name,lag=lag,alpha=alpha,failures=failures,wall_seconds=time.perf_counter()-begin,**metrics(pred,truth));record['arrays']=save_array('free_sim_'+name+'_v1',predictions=pred);dump(OUT/('free_sim_'+name+'_v1_result.json'),record);classical.append(record);print(json.dumps(dict(arm=name,status=record['status'],metrics=record['metrics'])),flush=True)
 import open_loop71_v1 as native
 fits=[neural_fit(native,f,train,train_edges,histories,truth,val_edges) for f in campaign['fits']]
 obj=dict(utc=utc(),status='DEVELOPMENT_FREE_SIMULATION_ASSESSMENT_CLOSED',fits=fits,classical=classical,protocol_sha256=sha(ROOT/'FREE_SIMULATION_PROTOCOL71_V1.md'),calibration_sha256=sha(cal_path),campaign_sha256=sha(OUT/'CAMPAIGN71_V3.json'),val_rows=256,history_steps=49,free_future_steps=50,reserved_eval_rows_accessed=0,wall_seconds=time.perf_counter()-start,new_method=False,unseen_structural_composition_test=False,accepted_final_directions=0)
 dump(OUT/'FREE_SIMULATION_ASSESSMENT71_V1.json',obj);print(json.dumps(dict(status=obj['status'],wall_seconds=obj['wall_seconds'])),flush=True)

if __name__=='__main__':main()
