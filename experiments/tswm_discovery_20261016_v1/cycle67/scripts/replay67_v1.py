"""Independent raw-source reconstruction, point metrics, ridge SVD and native replay."""
import sys,pathlib,math,statistics,csv,json,hashlib,datetime,collections
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from engine67_v2 import Engine,torch
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs'
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
counts=collections.Counter();errors={}
def compare(name,a,b,tol):
 a=np.asarray(a,dtype='float64');b=np.asarray(b,dtype='float64');err=float(np.max(np.abs(a-b)));errors[name]=max(errors.get(name,0.),err);assert err<=tol,(name,err,tol)
assert read(OUT/'campaign_state67_v1.json')['status']=='VALIDATION_SELECTION_AND_EVALUATION_TERMINAL_SUCCESS'
cm=read(OUT/'CASE_MANIFEST67_V1.json');data={};source=read(OUT/'DATA_SOURCE67_V4.json');raw={}
for row in source['files']:
 p=pathlib.Path(row['path']);assert sha(p)==row['sha256'];counts['raw_source_hashes']+=1
 with p.open(encoding='utf-8-sig',newline='') as f:r=csv.DictReader(f);r.fieldnames=[n.strip() for n in r.fieldnames];rr=list(r)
 def number(v):
  try:return float(v)
  except (ValueError,TypeError):return float('nan')
 raw[p.name]=np.array([[number(r[n]) for n in ['Libre GL','Carbs','Protein','Fat']] for r in rr],dtype='float64').T
for part,rec in cm['subsets'].items():
 assert sha(rec['path'])==rec['sha256'];data[part]=np.load(rec['path'],allow_pickle=False);z=data[part]
 for i,case in enumerate(rec['cases']):
  o=case['origin'];a=raw[case['participant']];hist=a[:,o-2880:o].copy();ix=np.arange(2880);last=np.maximum.accumulate(np.where(np.isfinite(hist[0]),ix,-100000));missing=~np.isfinite(hist[0]);allowed=missing&(ix-last<=15)&(last>=0);hist[0,allowed]=hist[0,last[allowed]];hist[1:]=np.nan_to_num(hist[1:],nan=0);u=np.nan_to_num(a[1:,o:o+720],nan=0).astype('float32');y=a[0,o:o+720].astype('float32')
  assert np.array_equal(hist.astype('float32'),z['history'][i]) and np.array_equal(u,z['known'][i]) and np.array_equal(y,z['target'][i],equal_nan=True) and np.array_equal(np.isfinite(y),z['mask'][i]);counts['processed_cases_from_source']+=1
  yy=z['history'][i,0].astype('float64');mu=statistics.mean(yy);sd=statistics.stdev(yy);sd=(sd or 1.)+1e-8;features=np.r_[[statistics.mean(block) for block in np.array_split((yy-mu)/sd,96)],[math.fsum(block) for rowu in u for block in np.array_split(rowu.astype('float64'),12)]];compare('source_features',features,z['features'][i],1e-9)
fit=read(OUT/'BASELINE_FIT67_V1.json');assert sha(fit['artifact'])==fit['sha256'];bank=np.load(fit['artifact'],allow_pickle=False);train=data['TRAIN'];X=(train['features']-bank['xmean'])/bank['xsd'];compare('train_feature_bank',X,bank['Z'],0)
Y=np.empty(train['target'].shape,dtype='float64')
for i,hist in enumerate(train['history']):mu=statistics.mean(hist[0].astype('float64'));sd=(statistics.stdev(hist[0].astype('float64')) or 1.)+1e-8;Y[i]=(train['target'][i].astype('float64')-mu)/sd
assert np.array_equal(train['mask'],bank['mask']);valid=train['mask'];compare('train_target_bank',Y[valid],bank['Y'][valid],1e-10);groups=collections.defaultdict(list)
for j in range(720):groups[valid[:,j].tobytes()].append(j)
for maskbytes,cols in groups.items():
 good=np.frombuffer(maskbytes,dtype=np.bool_);x=X[good];y=Y[np.ix_(good,cols)];xm=x.mean(0);ym=y.mean(0);U,s,Vh=np.linalg.svd(x-xm,full_matrices=False)
 for alpha in [1,100,10000]:
  coef=Vh.T@((s/(s*s+alpha))[:,None]*(U.T@(y-ym)));intercept=ym-xm@coef;compare('ridge_SVD_coefficients',np.vstack([coef,intercept]),bank['ridge_'+str(alpha)][:,cols],1e-7);counts['ridge_mask_group_checks']+=1
selection=read(OUT/'VALIDATION_SELECTION67_V1.json');selection_sha=sha(OUT/'VALIDATION_SELECTION67_V1.json');phase_results={}
for part in ['VALIDATION','EVALUATION']:
 result=read(OUT/(part+'_RESULTS67_V1.json'));phase_results[part]=result
 if part=='EVALUATION':assert result['selection_sha256']==selection_sha
 z=data[part];case_index={c['id']:i for i,c in enumerate(cm['subsets'][part]['cases'])};all_rows=[]
 for rec in result['case_receipts']:
  assert sha(rec['path'])==rec['sha256'];receipt=read(rec['path']);assert sha(receipt['artifact'])==receipt['artifact_sha256'];counts['case_forecast_archives']+=1;predictions=np.load(receipt['artifact'],allow_pickle=False);i=case_index[receipt['case']['id']];hist=z['history'][i];known=z['known'][i];truth=z['target'][i];mask=z['mask'][i];hy=hist[0].astype('float64');mu=statistics.mean(hy);sd=(statistics.stdev(hy) or 1.)+1e-8;q=(z['features'][i]-bank['xmean'])/bank['xsd'];dist=[math.fsum(float(a-b)**2 for a,b in zip(x,q)) for x in X];neighbors=sorted(range(len(X)),key=lambda n:(dist[n],n))
  for row in receipt['rows']:
   pred=predictions[row['array_key']];assert pred.shape==(720,) and np.isfinite(pred).all();d=[float(a)-float(b) for a,b in zip(pred[mask],truth[mask])];mae=math.fsum(abs(x) for x in d)/len(d);rmse=math.sqrt(math.fsum(x*x for x in d)/len(d));compare('MAE_mg_dL',mae,row['scores']['MAE_mg_dL'],1e-9);compare('RMSE_mg_dL',rmse,row['scores']['RMSE_mg_dL'],1e-9);compare('normalized_MAE',mae/sd,row['scores']['context_normalized_MAE'],1e-10);counts['forecast_metric_rows']+=1;method=row['method']
   if method.startswith('ridge_alpha'):
    alpha=int(method.removeprefix('ridge_alpha'));expected=np.r_[q,1]@bank['ridge_'+str(alpha)]*sd+mu;compare('ridge_prediction',pred,expected,1e-8);counts['classical_forecasts_replayed']+=1
   elif method.startswith('analog_k'):
    k=int(method.removeprefix('analog_k'));ix=neighbors[:k];yy=[]
    for j in range(720):values=[float(Y[n,j]) for n in ix if valid[n,j]];yy.append(mu+sd*(math.fsum(values)/len(values) if values else 0.))
    compare('analog_prediction',pred,yy,1e-8);counts['classical_forecasts_replayed']+=1
   elif method in ['context_mean','last_value','seasonal_day']:
    expected=np.full(720,mu) if method=='context_mean' else np.full(720,hist[0,-1]) if method=='last_value' else hist[0,-1440:-720];compare('sanity_prediction',pred,expected,1e-8);counts['classical_forecasts_replayed']+=1
  all_rows.extend(receipt['rows'])
 assert len(all_rows)==len(result['rows'])==result['prediction_rows'];counts['subset_result_rows']+=len(all_rows)
 for group in result['groups']:
  pr=[]
  for person in group['participant_means']:
   rr=[r for r in all_rows if r['participant']==person['participant'] and r['method']==group['method']];value=math.fsum(r['scores']['context_normalized_MAE'] for r in rr)/len(rr);compare('participant_mean',value,person['mean_normalized_MAE'],1e-10);pr.append(value)
  compare('group_mean',math.fsum(pr)/len(pr),group['mean_normalized_MAE'],1e-10)
val_lookup={g['method']:g['mean_normalized_MAE'] for g in phase_results['VALIDATION']['groups']};assert selection['analog_k']==min([1,4,16],key=lambda k:val_lookup['analog_k'+str(k)]) and selection['ridge_alpha']==min([1,100,10000],key=lambda a:val_lookup['ridge_alpha'+str(a)]);counts['validation_only_selections']=2
engine=Engine();rec=cm['subsets']['EVALUATION'];z=data['EVALUATION'];engine.set_case(z['history'][0],z['known'][0]);saved=read(OUT/'case_receipts'/('EVALUATION_'+rec['cases'][0]['id']+'.json'));p=np.load(saved['artifact'],allow_pickle=False)
for method,mc in [('native_predict',2021),('bayes_action_full_view',2021)]:pred,_=engine.forecast(method,mc);compare('fresh_native_'+method,p[method+'_MC'+str(mc)],pred,1e-4);counts['fresh_native_forecasts']=counts.get('fresh_native_forecasts',0)+1
obj=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='PASS',counts=dict(counts),maximum_absolute_errors=errors,case_manifest_sha256=sha(OUT/'CASE_MANIFEST67_V1.json'),validation_results_sha256=sha(OUT/'VALIDATION_RESULTS67_V1.json'),selection_sha256=selection_sha,evaluation_results_sha256=sha(OUT/'EVALUATION_RESULTS67_V1.json'),scope='All490 retained data cases reconstructed from raw CSVs; all forecast rows independently scored; supervised ridge SVD and classical predictions replayed; two fixed fresh native EVAL calls. No causal validation.',accepted_final_directions=0)
with (OUT/'INDEPENDENT_REPLAY67_V1.json').open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,allow_nan=False)
print(json.dumps(obj),flush=True)
