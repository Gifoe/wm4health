import sys,pathlib
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from engine67_v2 import *
from baselines67_v1 import Baselines
assert read(OUT/'NATIVE_QUALIFICATION67_V1.json')['status']=='PASS';cm=read(OUT/'CASE_MANIFEST67_V1.json');root=artifact_root();receipt_dir=OUT/'case_receipts';receipt_dir.mkdir(exist_ok=True);native_methods=['native_predict','clamped_simulate']+MODES
scripts=['common67_v1.py','engine67_v2.py','baselines67_v1.py','campaign67_v1.py'];identity=dict(protocol_sha256=sha(ROOT/'PROTOCOL67_V1.md'),case_manifest_sha256=sha(OUT/'CASE_MANIFEST67_V1.json'),source_sha256={n:sha(ROOT/'scripts'/n) for n in scripts},qualification_sha256=sha(OUT/'NATIVE_QUALIFICATION67_V1.json'),model_source_receipt_sha256=sha(PROJECT/'cycle66'/'outputs'/'MODEL_FILES66_V1.json'))
if not (OUT/'EXECUTION67_V1.json').exists():dump(OUT/'EXECUTION67_V1.json',dict(utc=utc(),**identity))
else:
 old=read(OUT/'EXECUTION67_V1.json');assert all(old[k]==v for k,v in identity.items())
registry_sha=sha(OUT/'EXECUTION67_V1.json');data={}
for part in ['TRAIN','VALIDATION']:
 rec=cm['subsets'][part];assert sha(rec['path'])==rec['sha256'];data[part]=np.load(rec['path'],allow_pickle=False)
base=Baselines()
if (OUT/'BASELINE_FIT67_V1.json').exists():
 rec=read(OUT/'BASELINE_FIT67_V1.json');assert sha(rec['artifact'])==rec['sha256'];z=np.load(rec['artifact'],allow_pickle=False)
 for name in ['xmean','xsd','Z','Y','mask']:setattr(base,name,z[name])
 base.coef={a:z['ridge_'+str(a)] for a in [1,100,10000]}
else:base.fit(data['TRAIN'])
engine=Engine()
def aggregate(rows):
 groups=[]
 for method in sorted(set(r['method'] for r in rows)):
  rr=[r for r in rows if r['method']==method];people=[]
  for name in sorted(set(r['participant'] for r in rr)):
   pr=[r for r in rr if r['participant']==name];people.append(dict(participant=name,cases=len(set(r['case'] for r in pr)),mean_normalized_MAE=float(np.mean([r['scores']['context_normalized_MAE'] for r in pr])),MAE_mg_dL=float(np.mean([r['scores']['MAE_mg_dL'] for r in pr])),RMSE_mg_dL=float(np.mean([r['scores']['RMSE_mg_dL'] for r in pr]))))
  groups.append(dict(method=method,participants=len(people),case_predictions=len(rr),mean_normalized_MAE=float(np.mean([r['mean_normalized_MAE'] for r in people])),MAE_mg_dL=float(np.mean([r['MAE_mg_dL'] for r in people])),RMSE_mg_dL=float(np.mean([r['RMSE_mg_dL'] for r in people])),participant_means=people))
 return groups
def run_subset(part,selection=None):
 marker=OUT/(part+'_RESULTS67_V1.json')
 if marker.exists():return read(marker)
 if part not in data:
  rec=cm['subsets'][part];assert sha(rec['path'])==rec['sha256'];data[part]=np.load(rec['path'],allow_pickle=False)
 selected_sha=sha(OUT/'VALIDATION_SELECTION67_V1.json') if selection else None;rows=[];receipts=[];start=time.perf_counter();z=data[part]
 for i,case in enumerate(cm['subsets'][part]['cases']):
  rp=receipt_dir/(part+'_'+case['id']+'.json');ap=root/'predictions'/(part+'_'+case['id']+'.npz');ap.parent.mkdir(exist_ok=True)
  if rp.exists():
   rr=read(rp);assert rr['execution_sha256']==registry_sha and rr['selection_sha256']==selected_sha and sha(rr['artifact'])==rr['artifact_sha256'];rows.extend(rr['rows']);receipts.append(dict(path=str(rp),sha256=sha(rp)));continue
  assert not ap.exists(),'Orphaned case arrays; inspect before resuming';state('campaign_state67_v1.json',dict(status='CASE_LIVE',split=part,case=case['id'],completed_cases=i,total_cases=len(cm['subsets'][part]['cases'])));history,known,target,mask=(z[n][i] for n in ['history','known','target','mask']);preds={};case_rows=[];classical=base.predict(history,known)
  if selection:classical={k:v for k,v in classical.items() if k in ['context_mean','last_value','seasonal_day','analog_k'+str(selection['analog_k']),'ridge_alpha'+str(selection['ridge_alpha'])]}
  def keep(method,mc,pred,timing):
   key=method+'_MC'+str(mc);preds[key]=pred;case_rows.append(dict(split=part,case=case['id'],participant=case['participant'],method=method,MC_seed=mc,array_key=key,scores=metric(pred,target,mask,history),timing=timing))
  for method,pred in classical.items():keep(method,None,pred,dict(scope='Classical inference; no end-to-end standalone latency comparison'))
  engine.set_case(history,known)
  for method in native_methods:
   for mc in ([2021,2022,2023] if method in MODES else [2021]):
    pred,timing=engine.forecast(method,mc);keep(method,mc,pred,timing)
  temporary=ap.with_suffix('.tmp')
  with temporary.open('xb') as f:np.savez_compressed(f,**preds)
  assert not ap.exists();temporary.replace(ap);rr=dict(utc=utc(),status='CASE_CLOSED',case=case,artifact=str(ap),artifact_sha256=sha(ap),execution_sha256=registry_sha,selection_sha256=selected_sha,rows=case_rows);dump(rp,rr);rows.extend(case_rows);receipts.append(dict(path=str(rp),sha256=sha(rp)));print(json.dumps(dict(split=part,completed_cases=i+1,total_cases=len(cm['subsets'][part]['cases']),predictions=len(rows))),flush=True)
 obj=dict(utc=utc(),status='FROZEN_SUBSET_SCORED',split=part,participants=len(set(r['participant'] for r in rows)),cases=len(cm['subsets'][part]['cases']),prediction_rows=len(rows),rows=rows,groups=aggregate(rows),case_receipts=receipts,selection_sha256=selected_sha,execution_sha256=registry_sha,elapsed_seconds=time.perf_counter()-start)
 dump(marker,obj);return obj
val=run_subset('VALIDATION')
sp=OUT/'VALIDATION_SELECTION67_V1.json'
if sp.exists():selection=read(sp)
else:
 lookup={g['method']:g['mean_normalized_MAE'] for g in val['groups']};k=min([1,4,16],key=lambda k:lookup['analog_k'+str(k)]);alpha=min([1,100,10000],key=lambda a:lookup['ridge_alpha'+str(a)]);selection=dict(utc=utc(),status='FROZEN_BEFORE_EVALUATION',analog_k=k,ridge_alpha=alpha,validation_results_sha256=sha(OUT/'VALIDATION_RESULTS67_V1.json'),primary_metric='participant-mean context-normalized MAE',native_arms_changed=False,EVALUATION_scored=False);dump(sp,selection)
assert selection['validation_results_sha256']==sha(OUT/'VALIDATION_RESULTS67_V1.json');test=run_subset('EVALUATION',selection);state('campaign_state67_v1.json',dict(status='VALIDATION_SELECTION_AND_EVALUATION_TERMINAL_SUCCESS',validation_predictions=val['prediction_rows'],evaluation_predictions=test['prediction_rows'],EVAL_selection_sha256=sha(sp)));print(json.dumps(dict(status='VALIDATION_SELECTION_AND_EVALUATION_TERMINAL_SUCCESS',validation_predictions=val['prediction_rows'],evaluation_predictions=test['prediction_rows'])),flush=True)
