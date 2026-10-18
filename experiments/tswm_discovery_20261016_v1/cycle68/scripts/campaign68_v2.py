import sys,pathlib
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from native68_v1 import *
from operators68_v1 import *
import time
assert read(OUT/'QUALIFICATION68_V1.json')['status']=='PASS'
manifest=read(OUT/'CASE_MANIFEST68_V1.json');assert manifest['protocol_sha256']==sha(ROOT/'PROTOCOL68_V1.md');native=NormWear();meta=MetaID();base=storage('artifact')/'predictions';base.mkdir(exist_ok=True);all_rows=[];ledger=[];diagnostics=[]
def score_case(case,subset,selected=None):
 assert sha(case['path'])==case['sha256'];z=np.load(case['path']);history=z['history'];known=z['known'];target=z['target'];native.set_case(history,known);arrays={};rows=[];start=time.perf_counter()
 def add(method,pred,**extra):
  key=method+'__'+str(extra.get('MC_seed',2021));pred=np.asarray(pred);assert pred.shape==(H,)
  if np.isfinite(pred).all():rows.append(dict(case=case['id'],subset=subset,family=case['family'],method=method,status='FINITE',**metric(pred,target,history),**extra))
  else:rows.append(dict(case=case['id'],subset=subset,family=case['family'],method=method,status='NONFINITE',**extra))
  arrays[key]=pred
 for mode in ['native_predict','clamped_simulate']:add(mode,native.forecast(mode))
 for mode in ['bayes_indexed','bayes_same_view']:
  for mc in [2021,2022,2023]:add(mode,native.forecast(mode,mc),MC_seed=mc)
 mp,aux=meta.forecast(history,known);add('meta_id_author_ensemble',mp);add('context_mean',np.repeat(history[0].mean(),H));add('last_value',np.repeat(history[0,-1],H))
 for degree in [1,3]:
  alphas=[1e-8,1e-4,.1] if subset=='VAL' else [selected['raw'][str(degree)]]
  for alpha in alphas:
   method=f'raw_arx_d{degree}_a{alpha:g}';pred,fit=raw_forecast(history,known,degree,alpha);add(method,pred,degree=degree,alpha=alpha)
   for key,v in fit.items():arrays[method+'__fit_'+key]=v
 configs=[(k,a) for k in [4,16,32] for a in [.001,.1,10.]] if subset=='VAL' else [(selected['latent']['k'],selected['latent']['alpha'])]
 recon=native.reconstruction();diag=dict(case=case['id'],native_context_reconstruction_MAE=float(abs(recon-history[0]).mean()),native_context_reconstruction_normalized_MAE=float(abs(recon-history[0]).mean()/norm(history)[1][0,0]),meta_aux=aux,latent=[])
 for k,alpha in configs:
  method=f'latent_ridge_k{k}_a{alpha:g}';future,fit=latent_forecast(native.E,history,known,k,alpha);pred=native.decode(future);add(method,pred,k=k,alpha=alpha)
  projected=fit['context_coefficients']@fit['basis']+fit['mean'];p=native.reconstruction(projected);diag['latent'].append(dict(k=k,alpha=alpha,context_projection_decode_normalized_MAE=float(abs(p-history[0]).mean()/norm(history)[1][0,0])));arrays[method+'__basis']=fit['basis'];arrays[method+'__mean']=fit['mean']
  for key,v in fit['operator'].items():arrays[method+'__operator_'+key]=v
 ap=base/(case['id']+'.npz');assert not ap.exists();np.savez_compressed(ap,**arrays);rec=dict(case=case['id'],subset=subset,family=case['family'],path=str(ap),sha256=sha(ap),arrays={key:list(value.shape) for key,value in arrays.items()},forecast_count=len(rows),seconds=time.perf_counter()-start);ledger.append(rec);diagnostics.append(diag);all_rows.extend(rows);dump(OUT/'case_receipts'/(case['id']+'.json'),dict(utc=utc(),source=case,artifact=rec,metrics=rows,diagnostics=diag));state('campaign_state68_v2.json',dict(status='FORMAL_CASE_COMPLETE',subset=subset,case=case['id'],completed_cases=len(ledger),score_rows=len(all_rows)));print(json.dumps(dict(case=case['id'],predictions=len(rows),seconds=rec['seconds'])),flush=True)
def aggregate(rows):
 methods=sorted({r['method'] for r in rows});result=[]
 for method in methods:
  selected=[r for r in rows if r['method']==method]
  if any(r['status']!='FINITE' for r in selected):result.append(dict(method=method,status='DISQUALIFIED_NONFINITE'));continue
  cases=sorted({r['case'] for r in selected});units=[]
  for case in cases:
   rec=[r for r in selected if r['case']==case];units.append(dict(case=case,family=rec[0]['family'],**{m:float(np.mean([v[m] for v in rec])) for m in ['normalized_MAE','MAE','RMSE','first50_normalized_MAE','last50_normalized_MAE']}))
  means={m:float(np.mean([np.mean([v[m] for v in units if v['family']==f]) for f in ['linear','cubic']])) for m in ['normalized_MAE','MAE','RMSE','first50_normalized_MAE','last50_normalized_MAE']};result.append(dict(method=method,status='FINITE',cases=len(cases),rows=len(selected),**means,per_case=units,per_family={f:{m:float(np.mean([v[m] for v in units if v['family']==f])) for m in ['normalized_MAE','MAE','RMSE']} for f in ['linear','cubic']}))
 return result
for case in [r for r in manifest['cases'] if r['subset']=='VAL']:score_case(case,'VAL')
val=aggregate(all_rows);dump(OUT/'VALIDATION_RESULTS68_V1.json',dict(utc=utc(),rows=all_rows,aggregate=val))
good=[r for r in val if r['status']=='FINITE'];lm=min([r for r in good if r['method'].startswith('latent_ridge_')],key=lambda r:(r['normalized_MAE'],r['method']));sample=next(r for r in all_rows if r['method']==lm['method']);selected=dict(latent=dict(k=sample['k'],alpha=sample['alpha'],method=lm['method']),raw={})
for degree in [1,3]:
 winner=min([r for r in good if r['method'].startswith(f'raw_arx_d{degree}_')],key=lambda r:(r['normalized_MAE'],r['method']));sample=next(r for r in all_rows if r['method']==winner['method']);selected['raw'][str(degree)]=sample['alpha']
dump(OUT/'VALIDATION_SELECTION68_V1.json',dict(utc=utc(),status='VAL_ONLY_SELECTION_FROZEN_BEFORE_EVAL',selected=selected,validation_results_sha256=sha(OUT/'VALIDATION_RESULTS68_V1.json'),evaluated_test_scores_before_selection=False));selection_sha=sha(OUT/'VALIDATION_SELECTION68_V1.json');before=len(all_rows)
for case in [r for r in manifest['cases'] if r['subset']=='EVAL']:
 assert sha(OUT/'VALIDATION_SELECTION68_V1.json')==selection_sha;score_case(case,'EVAL',selected)
dump(OUT/'EVALUATION_RESULTS68_V1.json',dict(utc=utc(),rows=all_rows[before:],aggregate=aggregate(all_rows[before:]),selection_sha256=selection_sha))
dump(OUT/'FORECAST_LEDGER68_V1.json',dict(utc=utc(),cases=ledger,diagnostics=diagnostics,source_manifest_sha256=sha(OUT/'CASE_MANIFEST68_V1.json'),validation_selection_sha256=selection_sha,native_model_source_sha256=sha(PROJECT/'cycle66'/'outputs'/'MODEL_FILES66_V1.json'),meta_model_source_sha256=sha(OUT/'MODEL_FILES68_V1.json'),native_parameters=sum(p.numel() for p in native.model.parameters()),meta_parameters=meta.parameter_count,neural_fits=0,score_rows=len(all_rows),raw_data_and_forecast_storage='Server G only'))
state('campaign_state68_v2.json',dict(status='VALIDATION_AND_EVALUATION_TERMINAL_SUCCESS',completed_cases=len(ledger),score_rows=len(all_rows),selection_sha256=selection_sha));print('VALIDATION_AND_EVALUATION_TERMINAL_SUCCESS',flush=True)
