import sys,pathlib,math,statistics,collections,json
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from native69_v1 import *
from scipy.io import loadmat
assert read(OUT/'campaign69_v1_exit.json')['exit_code']==0;manifest=read(OUT/'DATA_MANIFEST69_V1.json');results=read(OUT/'RESULTS69_V1.json');ledger=read(OUT/'FORECAST_LEDGER69_V1.json');source=read(manifest['source_receipt']);assert sha(manifest['source_receipt'])==manifest['source_receipt_sha256'];counts=collections.Counter();errors={}
def check(name,a,b,tol):
 err=float(np.max(np.abs(np.asarray(a)-np.asarray(b))));errors[name]=max(err,errors.get(name,0.));assert err<=tol,(name,err,tol)
for row in source['sources']:
 raw=pathlib.Path(row['server_path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==row['sha256'];assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==row['git_blob_sha1'];counts['upstream_source_and_checkpoint_blobs']+=1
for row in source['data']:assert sha(row['server_path'])==row['sha256'];counts['numeric_source_hashes']+=1
items=[]
for case in manifest['cases']:
 assert sha(case['path'])==case['sha256'];z={k:v for k,v in np.load(case['path'],allow_pickle=False).items()};raw=loadmat(case['source_path'])
 if case['benchmark']=='WH':u=raw['uBenchMark'].ravel()[5200:184000];y=raw['yBenchMark'].ravel()[5200:184000];reference=dict(train_u=u[:100000],train_y=y[:100000],test_u=u[100000:],test_y=y[100000:])
 elif case['benchmark']=='Tanks':reference=dict(train_u=raw['uEst'].ravel(),train_y=raw['yEst'].ravel(),test_u=raw['uVal'].ravel(),test_y=raw['yVal'].ravel())
 else:
  u=raw[['u11','u12'][case['index']]].ravel();y=raw[['z11','z12'][case['index']]].ravel();reference=dict(train_u=u[:400],train_y=y[:400],test_u=u[400:],test_y=y[400:])
 for key,value in reference.items():check('independent_split_values',value,z[key],0)
 counts['independently_reconstructed_trajectories']+=1;items.append(dict(case=case,arrays=z))
item_map={x['case']['case']:x for x in items};meta=Meta();dyno=Dyno()
for rec in ledger['records']:
 assert sha(rec['path'])==rec['sha256'];archive=np.load(rec['path'],allow_pickle=False);item=item_map[rec['case']];data=item['arrays'];case=item['case'];n=rec['initial'];true=data['test_y'][n:];native=archive['native_normalized_means'].astype('float64');sd=archive['native_normalized_stds'].astype('float64');assert native.shape==sd.shape==(rec['native_contexts'],len(true)) and np.isfinite(native).all() and np.isfinite(sd).all() and (sd>0).all();assert rec['context_starts']==list(range(0,case['context_length']-400+1,case['context_stride']));mu=rec['normalization']['y_mean'];scale=rec['normalization']['y_sd'];expected={}
 for mode in ['mean','precision']:
  weights=None if mode=='mean' else 1/(sd*sd);m=np.average(native,axis=0,weights=weights);weights=np.ones_like(sd)/len(sd) if mode=='mean' else weights/weights.sum(0);est=np.sqrt(np.sum(weights**2*sd**2,axis=0));mix=np.sqrt(np.average(sd*sd+native*native,axis=0,weights=weights)-m*m);expected['meta_'+mode]=m*scale+mu;check('ensemble_'+mode,expected['meta_'+mode],archive['meta_'+mode],1e-9);check('estimator_sd_'+mode,est*scale,archive['meta_'+mode+'_estimator_sd'],1e-9);check('mixture_moment_sd_'+mode,mix*scale,archive['meta_'+mode+'_mixture_moment_sd'],1e-8);counts['independently_recomputed_ensembles']+=1
 pred,_=dyno.forecast(data,dict(case,official_initial=n),items);check('fresh_full_dyno',pred[n:],archive['dyno_author'],0);counts['fresh_dyno_full_trajectories']+=1;check('context_mean',archive['context_mean'],np.repeat(statistics.mean(data['train_y']),len(true)),1e-10)
 rows=[r for r in results['rows'] if r['record']==rec['record']]
 for row in rows:
  p=archive[row['method']];assert p.shape==true.shape and np.isfinite(p).all();d=[float(a)-float(b) for a,b in zip(p,true)];rmse=math.sqrt(math.fsum(q*q for q in d)/len(d));mae=math.fsum(abs(q) for q in d)/len(d);check('RMSE',rmse,row['RMSE'],1e-9);check('MAE',mae,row['MAE'],1e-9);check('normalized_MAE',mae/statistics.stdev(data['train_y']),row['training_normalized_MAE'],1e-9);common=max(case['official_initial'],n)-n;check('common_suffix_RMSE',math.sqrt(math.fsum(q*q for q in d[common:])/len(d[common:])),row['common_suffix_RMSE'],1e-9);half=len(d)//2
  for name,arr in [('early_half_RMSE',d[:half]),('late_half_RMSE',d[half:]),('first100_RMSE',d[:100]),('last100_RMSE',d[-100:])]:check(name,math.sqrt(math.fsum(q*q for q in arr)/len(arr)),row[name],1e-9)
  if row['method'].startswith('meta_'):
   for field,key in [('estimator_three_sd_fraction','estimator_sd'),('mixture_moment_three_sd_fraction','mixture_moment_sd')]:value=sum(abs(q)<=3*s for q,s in zip(d,archive[row['method']+'_'+key]))/len(d);check(field,value,row[field],0)
  counts['full_trajectory_metric_rows']+=1;counts['scored_point_records']+=len(d)
 meta.set_context(data,case);tu,iu,iy=meta.init_state(data,n,case['benchmark']);fu=tu[n:n+100]
 if len(fu)<100:fu=torch.cat([fu,fu[-1:].expand(100-len(fu))])
 m,s=meta.query(iu,iy,fu);valid=min(100,len(true));check('fresh_native_first_mean',m[:,:valid,0].cpu().numpy(),archive['native_normalized_means'][:,:valid],1e-6);check('fresh_native_first_std',s[:,:valid,0].cpu().numpy(),archive['native_normalized_stds'][:,:valid],1e-6);counts['fresh_meta_native_queries']+=1
 if case['benchmark']=='WH':
  offset=100*(len(rec['native_queries'])//2);t=n+offset;iu=tu[t-30:t][None,:,None].expand(meta.contexts,-1,-1);iy=torch.tensor(archive['native_normalized_means'][:,offset-30:offset,None],device='cuda');m,s=meta.query(iu,iy,tu[t:t+100]);check('fresh_native_midrun_mean',m[:,:,0].cpu().numpy(),archive['native_normalized_means'][:,offset:offset+100],1e-6);check('fresh_native_midrun_std',s[:,:,0].cpu().numpy(),archive['native_normalized_stds'][:,offset:offset+100],1e-6);counts['fresh_meta_native_queries']+=1
 counts['forecast_archive_hashes']+=1;print(rec['record'],'REPLAYED',flush=True)
dump(OUT/'INDEPENDENT_REPLAY69_V1.json',dict(utc=utc(),status='PASS',counts=dict(counts),maximum_absolute_errors=errors,results_sha256=sha(OUT/'RESULTS69_V1.json'),ledger_sha256=sha(OUT/'FORECAST_LEDGER69_V1.json'),manifest_sha256=sha(OUT/'DATA_MANIFEST69_V1.json'),scope='All official splits manually reconstructed, all whole-trajectory metrics and ensemble moments recalculated; fresh native first/midrun calls and full author dynoNet outputs. No independent upstream training reproduction or joint predictive calibration claim.',accepted_final_directions=0));print(json.dumps(dict(status='PASS',counts=dict(counts),maximum_absolute_errors=errors)),flush=True)
