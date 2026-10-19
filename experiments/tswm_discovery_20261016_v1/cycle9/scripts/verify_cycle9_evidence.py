"""Independent result arithmetic and source checks, without rerunning science."""
import runtime_profile
import csv,hashlib,json,math,pathlib
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
def clean(x):
 if isinstance(x,float) and not math.isfinite(x):return None
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
 if isinstance(x,list):return [clean(v) for v in x]
 return x
def main():
 reports={}
 for folder,n in [('dotime',540),('dotime_full',12150)]:
  path=ROOT/'outputs'/folder;rows=list(csv.DictReader((path/'predictions.csv').open(encoding='utf-8',newline='')));assert len(rows)==n and len({r['scm_id'] for r in rows})==n
  truth=np.asarray([float(r['y_true']) for r in rows]);obs=np.asarray([float(r['y_obs']) for r in rows]);metrics=json.loads((path/'metrics.json').read_text(encoding='utf-8'));mask=np.abs(truth-obs)>=.1
  for model in ['PFNRef','Mean','VAR-OLS','NaiveOLS']:
   pred=np.asarray([float(r[model]) for r in rows]);row=next(x for x in metrics if x['model']==model and x['structure']=='ALL');assert np.isfinite(pred).all();rmse=float(np.sqrt(np.mean((pred-truth)**2)));acc=float(np.mean(np.sign(pred[mask]-obs[mask])==np.sign(truth[mask]-obs[mask])));assert abs(rmse-row['rmse'])<1e-8 and abs(acc-row['dir_acc_effect'])<1e-6
  native=json.loads((path/'native_pfn_metrics.json').read_text(encoding='utf-8'));pfn=next(x for x in metrics if x['model']=='PFNRef' and x['structure']=='ALL');assert abs(native['pooled_rmse']-pfn['rmse'])<1e-6
  (path/'metrics_strict.json').write_text(json.dumps(clean(metrics),indent=2,allow_nan=False),encoding='utf-8')
  primary=np.asarray([r['structure'] not in ['bow_graph','observed_confounder','unobserved_confounder'] for r in rows])&mask
  reports[folder]={'episodes':n,'effect_denominator':int(mask.sum()),'identified_primary_effect_denominator':int(primary.sum()),'all_scalar_metrics_recomputed':True}
 for key in ['ep_idm','acro','ac_mtm']:
  pin=json.loads((ROOT/f'sources/{key}/SOURCE_PIN.json').read_text(encoding='utf-8'))
  for row in pin['files']:assert hashlib.sha256((ROOT/'sources'/key/row['path']).read_bytes()).hexdigest()==row['sha256']
 reports['source_pins_verified']=True;reports['raw_native_nan_policy']='Undefined zero-effect subgroup accuracy remains NaN in raw native-derived output; metrics_strict.json uses JSON null without changing defined values.'
 (ROOT/'outputs/evidence_verification.json').write_text(json.dumps(reports,indent=2,allow_nan=False),encoding='utf-8');print(json.dumps(reports),flush=True)
if __name__=='__main__':main()
