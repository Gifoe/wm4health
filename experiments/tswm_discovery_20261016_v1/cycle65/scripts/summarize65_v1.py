from common65 import *
import numpy as np
e=read(OUT/'EVALUATION65_V1.json');f=read(OUT/'FITS65_V1.json');r=read(OUT/'INDEPENDENT_REPLAY65_V1.json');assert r['status']=='PASS';groups=[]
methods=sorted(set(x['method'] for x in e['laws']))
for mech in ['linear','nonlinear']:
 for pk in ['zero','sine']:
  for method in methods:
   rows=[x for x in e['laws'] if x['mechanism']==mech and x['query']==pk and x['method']==method]
   if not rows:continue
   entry=dict(mechanism=mech,query=pk,method=method,laws=len(rows),independent_training_seeds=len(set(x['training_seed'] for x in rows if x['training_seed'] is not None)),MC_seeds=len(set(x['MC_seed'] for x in rows)),scores={})
   for target in ['do','obs']:
    vals=[x['scores'][target]['normalized_finite_sliced_W1'] for x in rows];tail=[x['scores'][target]['terminal_tail_abs_error'] for x in rows];entry['scores'][target]=dict(mean_normalized_sliced_W1=float(np.mean(vals)),min_normalized_sliced_W1=min(vals),max_normalized_sliced_W1=max(vals),mean_terminal_tail_abs_error=float(np.mean(tail)),seed_means=[dict(training_seed=ts,mean=float(np.mean([x['scores'][target]['normalized_finite_sliced_W1'] for x in rows if x['training_seed']==ts]))) for ts in sorted(set(x['training_seed'] for x in rows if x['training_seed'] is not None))])
   times=[x['runtime_seconds'] for x in rows if x['runtime_seconds'] is not None];entry['mean_runtime_seconds']=None if not times else float(np.mean(times));groups.append(entry)
obj=dict(utc=utc(),status='EXECUTION_AND_REPLAY_CLOSED_ANALYSIS_PENDING',accepted_final_directions=0,groups=groups,fit_summaries=[{k:x[k] for k in ['key','model','parameters','optimizer_updates','target_scalar_exposures','elapsed_seconds','selected_checkpoint']} for x in f['fits']],models=f['models'],optimizer_updates=f['optimizer_updates'],unique_training_corpora=f['unique_training_corpora'],generated_laws=e['generated_laws'],scored_target_views=e['scored_target_views'],replay_max_scaled_errors=r['max_scaled_errors'],fit_elapsed_seconds=f['elapsed_seconds'],evaluation_elapsed_seconds=e['elapsed_seconds'])
dump(OUT/'SUMMARY65_V1.json',obj);print(json.dumps(dict(status=obj['status'],models=obj['models'],optimizer_updates=obj['optimizer_updates'],groups=groups)))
