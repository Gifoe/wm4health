"""Independent saved-prediction arithmetic and history-control replay; no neural re-search."""
import os
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
import pathlib,json,hashlib,datetime,csv,sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs';DEST=OUT/'free_simulation_closure_20261023_v1'
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  while b:=f.read(1048576):h.update(b)
 return h.hexdigest()
def write(p,j):
 with pathlib.Path(p).open('x',encoding='utf-8') as f:json.dump(j,f,indent=2,allow_nan=False)
exitp=OUT/'free_simulation71_recovery_20261023_v1_exit.json';assert read(exitp)['exit_code']==0
rp=OUT/'FREE_SIMULATION_ASSESSMENT71_V1.json';j=read(rp);assert j['status']=='DEVELOPMENT_FREE_SIMULATION_ASSESSMENT_CLOSED'
assert j['val_rows']==256 and j['history_steps']==49 and j['free_future_steps']==50 and j['reserved_eval_rows_accessed']==0
assert j['protocol_sha256']==sha(ROOT/'FREE_SIMULATION_PROTOCOL71_V1.md')=='261b702e630543728475a187c13f110b1a97a6bd2ae9af66089db35a1a0299c3'
assert sha(ROOT/'scripts/evaluate_free_simulation71_v3.py')=='2f09e76a7630e5f8cc63755001ff442628fa85574140d5b8c90cf915b5fdb0f7'
assert sha(ROOT/'scripts/evaluate_free_simulation71_v1.py')=='a7c0ffab12752976c72c45d06977edef0c4982d96686e703d2f6ba8f15153769'
cp=OUT/'BOUNDED_CAMPAIGN71_V1.json';campaign=read(cp);assert sha(cp)==j['campaign_sha256']
assert campaign['total_module_updates']==campaign['total_proposal_updates']==96000 and campaign['reserved_eval_rows_accessed']==0
calp=OUT/'HISTORY_CONTROL_TRAIN_CALIBRATION71_V1.json';cal=read(calp)
assert sha(calp)==j['calibration_sha256']=='01196c84f8982b4e70002c60e793079ed79a8cf2a958a5c03fe4b61cb5729dc0'
assert cal['val_rows_accessed']==cal['eval_rows_accessed']==0 and cal['train_rows']==2048
assert cal['implementation_sha256']==sha(ROOT/'scripts/history_controls71_v1.py')
for lag in (1,2):
 best=min((x for x in cal['grid_results'] if x['lag']==lag),key=lambda x:x['mse']);selected=next(x for x in cal['selected'] if x['lag']==lag)
 assert best['alpha']==selected['alpha'] and best['mse']==selected['mse']
acq=read(OUT/'ACQUISITION71_V1.json');data=pathlib.Path(acq['data_root'])
assert sha(data/'state_springs5_train.npy')==cal['source_data_sha256']
assert sha(data/'state_springs5_test.npy')=='2484c0a2f00528087cd59f392cd449173c95e513ced3efa033599ae625491e64'
assert sha(data/'edges_springs5_test.npy')=='9b7d18a19fa6ad8c37a9d1b30b83712c77cb440466901409bd7d619c154fcfc3'
train=np.load(data/'state_springs5_train.npy',mmap_mode='r',allow_pickle=False);dev=np.load(data/'state_springs5_test.npy',mmap_mode='r',allow_pickle=False)[:256].copy();edges=np.load(data/'edges_springs5_test.npy',mmap_mode='r',allow_pickle=False)[:256].copy()
assert train.shape==(2048,49,5,4) and dev.shape==(256,99,5,4) and edges.shape==(256,20)
history=dev[:,:49];truth=dev[:,49:];mu=np.asarray(cal['coordinate_mean']);scale=np.asarray(cal['coordinate_scale'])
assert np.allclose(train.reshape(-1,20).mean(axis=0),mu,rtol=2e-11,atol=2e-11)
assert np.allclose(np.maximum(train.reshape(-1,20).std(axis=0),1e-8),scale,rtol=2e-11,atol=2e-11)
assert [x['seed'] for x in j['fits']]==[2026711,2026712,2026713]
records=[];checks=[];trace_checks=[];receipts=[];predictions={};trajectory_scores={}
def check(a,b,label):
 err=float(np.max(abs(np.asarray(a)-np.asarray(b))));assert err<2e-11,(label,err);checks.append(dict(label=label,max_abs_error=err))
for fit in j['fits']:
 seed=fit['seed'];assert fit['module_updates']==fit['proposal_updates']==fit['reserved_eval_rows_accessed']==0
 cfit=next(x for x in campaign['fits'] if x['seed']==seed);fr=read(cfit['result_file'])
 assert sha(cfit['result_file'])==cfit['result_sha256'] and fr['epochs']==2000 and fr['module_updates']==fr['proposal_updates']==32000
 assert fit['checkpoint']==cfit['selected'] and sha(fit['checkpoint']['path'])==fit['checkpoint']['sha256']
 assert read(OUT/f'free_sim_seed{seed}_v1_result.json')==fit
 arms=fit['arms'];assert [x['arm'] for x in arms]==['encoder_argmax','native_search100','native_search1000','privileged_true_assignments']
 for a in arms:
  assert read(OUT/f"free_sim_seed{seed}_v1_{a['arm']}_result.json")==a
  orientation=a['orientation'];assert not orientation['val_labels_used_for_selection'] and orientation['flip']==int(orientation['native_reset_block_history_mse'][1]<orientation['native_reset_block_history_mse'][0])
  assert sha(a['arrays']['path'])==a['arrays']['sha256']
  with np.load(a['arrays']['path'],allow_pickle=False) as z:
   p=z['predictions'];g=z['assignments'];hm=z['history_mse']
  assert p.shape==(256,50,5,4) and g.shape==(256,20) and hm.shape==(256,) and np.isfinite(p).all() and np.isfinite(hm).all()
  check(hm.mean(),a['mean_native_history_mse'],f'{seed}/{a["arm"]}/history')
  check((g==(edges^orientation['flip'])).mean(),a['graph_assignment_accuracy'],f'{seed}/{a["arm"]}/graph')
  if a['arm']=='privileged_true_assignments':assert a['privileged'] and np.array_equal(g,edges^orientation['flip'])
  else:assert not a['privileged']
  records.append(a);predictions[(seed,a['arm'])]=p
 s100=next(x for x in arms if x['arm']=='native_search100')['search'];s1000=next(x for x in arms if x['arm']=='native_search1000')['search']
 assert [x['offset'] for x in s100]==[x['offset'] for x in s1000]==list(range(0,256,32))
 for a,b in zip(s100,s1000):
  assert a['steps']==100 and b['steps']==1000 and a['dummy_labels_only'] and b['dummy_labels_only']
  assert len(a['accepted_per_step'])==100 and len(b['accepted_per_step'])==1000
  equal=a['accepted_per_step']==b['accepted_per_step'][:100]
  trace_checks.append(dict(seed=seed,offset=a['offset'],acceptance_count_prefix_equal=equal))
for a in j['classical']:
 assert read(OUT/f"free_sim_{a['arm']}_v1_result.json")==a and not a['failures'] and not a['nonfinite_rows']
 assert sha(a['arrays']['path'])==a['arrays']['sha256']
 with np.load(a['arrays']['path'],allow_pickle=False) as z:p=z['predictions']
 assert p.shape==(256,50,5,4) and np.isfinite(p).all()
 records.append(a);predictions[(None,a['arm'])]=p
table=[]
for a in records:
 seed=a.get('seed');arm=a['arm'];p=predictions[(seed,arm)];assert a['status']=='FINITE_ALL_TRAJECTORIES' and not a['nonfinite_rows']
 per=np.mean((p.astype(np.float64)-truth)**2,axis=(1,2,3));trajectory_scores[(str(seed)+'_'+arm)]=per
 for expected in a['metrics']:
  H=expected['horizon'];ss=sa=ep=0.;cs=np.zeros(4);n=0;en=0
  for start in range(0,256,13):
   e=p[start:start+13,:H].astype(np.float64)-truth[start:start+13,:H]
   ss+=np.einsum('btnd,btnd->',e,e);sa+=abs(e).sum(dtype=np.float64);cs+=np.einsum('btnd,btnd->d',e,e);n+=e.size
   last=e[:,-1];ep+=np.einsum('bnd,bnd->',last,last);en+=last.size
  metrics=dict(prefix_mse=ss/n,prefix_mae=sa/n,endpoint_mse=ep/en,coordinate_mse=(cs/(n/4)).tolist())
  for key,value in metrics.items():check(value,expected[key],f'{seed}/{arm}/H{H}/{key}')
  table.append(dict(seed=seed,arm=arm,privileged=a.get('privileged',False),horizon=H,**metrics))
 receipts.append(dict(seed=seed,arm=arm,**a['arrays']))
# Independent augmented least squares on every presented history; same previously qualified bounds.
classical_replay=[]
for lag in (1,2):
 alpha=next(x['alpha'] for x in cal['selected'] if x['lag']==lag);ref=[]
 for h in history:
  z=(h.reshape(49,20)-mu)/scale;xx=[];yy=[]
  for t in range(lag-1,48):xx.append(np.concatenate([z[t-j] for j in range(lag)]+[np.ones(1)]));yy.append(z[t+1]-z[t])
  X=np.asarray(xx);Y=np.asarray(yy);pen=np.sqrt(len(X)*alpha)*np.eye(X.shape[1]);pen[-1,-1]=0
  coef=np.linalg.lstsq(np.vstack([X,pen]),np.vstack([Y,np.zeros((X.shape[1],20))]),rcond=None)[0]
  state=[x.copy() for x in z[-lag:]];forecast=[]
  for _ in range(50):
   phi=np.concatenate([state[-1-j] for j in range(lag)]+[np.ones(1)]);nxt=state[-1]+phi@coef;state.append(nxt);forecast.append((nxt*scale+mu).reshape(5,4))
  ref.append(forecast)
 ref=np.asarray(ref);saved=predictions[(None,'history_VAR'+str(lag))]
 err=float(abs(ref-saved).max());scaled=err/max(1.,float(abs(ref).max()));mse=float(np.mean((ref-truth)**2));stored=float(np.mean((saved-truth)**2));relative=abs(mse/stored-1)
 assert scaled<1e-7 and relative<1e-4,(lag,scaled,relative)
 classical_replay.append(dict(lag=lag,all_256_histories=True,prediction_max_abs_error=err,prediction_scaled_error=scaled,mse=mse,stored_mse=stored,relative_mse_difference=relative))
 assert np.array_equal(predictions[(None,'persistence')],np.repeat(history[:,-1:, :, :],50,axis=1))
summary=[]
for arm in ['encoder_argmax','native_search100','native_search1000','privileged_true_assignments','history_VAR1','history_VAR2','persistence']:
 r=[x for x in table if x['arm']==arm and x['horizon']==50];vals=np.asarray([x['prefix_mse'] for x in r]);q=dict(arm=arm,privileged=arm=='privileged_true_assignments',training_seeds=(3 if len(r)==3 else None),h50_mse_mean=float(vals.mean()),h50_mse_min=float(vals.min()),h50_mse_max=float(vals.max()))
 if len(r)==3:
  q['graph_accuracy_mean']=float(np.mean([x['graph_assignment_accuracy'] for x in records if x['arm']==arm]));q['h50_mse_seed_sd']=float(vals.std(ddof=1))
 summary.append(q)
var2=next(x['h50_mse_mean'] for x in summary if x['arm']=='history_VAR2');oracle=[x for x in table if x['arm']=='privileged_true_assignments' and x['horizon']==50]
obj=dict(status='C71_FULL_DEVELOPMENT_ASSESSMENT_INDEPENDENTLY_VERIFIED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),assessment_sha256=sha(rp),terminal_receipt=read(exitp),protocol_sha256=j['protocol_sha256'],calibration_sha256=sha(calp),campaign_sha256=sha(cp),source_revision=acq['revision'],source_repo=acq['repo'],training_module_updates=96000,training_proposal_updates=96000,evaluation_optimizer_updates=0,ordinary_neural_arms=9,privileged_diagnostic_arms=3,classical_controls=3,development_trajectories=256,reserved_eval_rows_accessed=0,scalar_checks=len(checks),max_scalar_error=max(x['max_abs_error'] for x in checks),table=table,summary=summary,search_prefix_checks=trace_checks,all_search_acceptance_prefixes_equal=all(x['acceptance_count_prefix_equal'] for x in trace_checks),independent_classical_replay=classical_replay,array_receipts=receipts,var2_h50_mse_lower_than_all_three_privileged_means=all(var2<x['prefix_mse'] for x in oracle),native_search_rerun=False,full_paper_reproduction=False,independent_structural_composition_test=False,accepted_final_directions=0,full_research_goal_achieved=False)
DEST.mkdir(exist_ok=False);write(DEST/'VERIFICATION71.json',obj)
st=read(ROOT.parent/'STORAGE_LOCATION_V1.json');aout=pathlib.Path(st['artifact_root'])/'cycle71/free_simulation_closure_20261023_v1';aout.mkdir(exist_ok=False);p=aout/'per_trajectory_h50_errors.npz';np.savez_compressed(p,**trajectory_scores)
write(DEST/'PER_TRAJECTORY_RECEIPT71.json',dict(path=str(p),sha256=sha(p),scope='Saved-prediction errors only; no new model or reserved EVAL'))
with (DEST/'METRICS71.csv').open('x',encoding='utf-8',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=['seed','arm','privileged','horizon','prefix_mse','prefix_mae','endpoint_mse']);writer.writeheader();writer.writerows({k:v for k,v in x.items() if k!='coordinate_mse'} for x in table)
print(json.dumps(dict(status=obj['status'],summary=summary,max_scalar_error=obj['max_scalar_error'],classical_replay=classical_replay,search_prefix_equal=obj['all_search_acceptance_prefixes_equal'],var2_beats_all_privileged_means=obj['var2_h50_mse_lower_than_all_three_privileged_means'])),flush=True)
