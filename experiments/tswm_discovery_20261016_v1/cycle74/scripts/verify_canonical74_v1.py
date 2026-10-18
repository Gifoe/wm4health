"""Independent companion-state rollout and all saved DEV score reconstruction."""
import sys,pathlib,time
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common74_v1 import *
import numpy as np
def companion(coef,u_hist,y_hist,future,lag):
 A=np.zeros((2*lag+1,2*lag+1));B=np.zeros(2*lag+1);B[0]=1.
 for i in range(1,lag):A[i,i-1]=1.
 A[lag,:lag]=coef[2:2+lag];A[lag,lag:2*lag]=coef[2+lag:];A[lag,-1]=coef[0];B[lag]=coef[1]
 for i in range(lag+1,2*lag):A[i,i-1]=1.
 A[-1,-1]=1.;s=np.r_[u_hist[-lag:][::-1],y_hist[-lag:][::-1],1.];out=[]
 for v in future:s=A@s+B*v;out.append(s[lag])
 return np.asarray(out)
def main():
 started=time.perf_counter();m=read(OUT/'TRAIN_DATA74_V2.json');r=read(OUT/'CANONICAL_ARX74_V2.json');exit_rec=read(OUT/'canonical74_v2_exit.json');assert exit_rec['exit_code']==0
 assert sha(OUT/'canonical74_v2_stdout.log')==exit_rec['log_sha256'];assert r['data_manifest_sha256']==sha(OUT/'TRAIN_DATA74_V2.json')
 normalized={};cases={x['case']:x for x in m['records']};checks=[];all_point_rows=0;max_target_difference=0.;max_companion_difference=0.
 for item in m['records']:
  assert sha(item['array_path'])==item['array_sha256'];a=np.load(item['array_path'],allow_pickle=False);sc=m['scaling'][item['dataset']];normalized[item['case']]=((a['u']-sc['mean'][0])/sc['std'][0],(a['y']-sc['mean'][1])/sc['std'][1])
 for row in r['rows']:
  assert sha(row['array_path'])==row['array_sha256'];p=np.load(row['array_path'],allow_pickle=False);errs=[]
  coef=np.load(row['model']['coefficient_path'],allow_pickle=False) if row['method'].startswith('ridge') else None
  if coef is not None:assert sha(row['model']['coefficient_path'])==row['model']['coefficient_sha256']
  for rec in row['per_record']:
   case=rec['case'];u,y=normalized[case];origins=p[case+'_origins'];target=p[case+'_target'];pred=p[case+'_pred'];assert list(origins)==cases[case]['dev_origins']
   actual=np.asarray([y[o:o+128] for o in origins]);d=float(abs(actual-target).max());assert d==0.;max_target_difference=max(max_target_difference,d)
   assert pred.shape==target.shape and np.isfinite(pred).all();error=pred-target;errs.append(error.ravel());all_point_rows+=pred.size
   for key,value in [('normalized_rmse',float(np.sqrt(np.mean(error**2)))),('normalized_mae',float(np.mean(abs(error))))]:assert abs(rec[key]-value)<1e-12,(case,key)
   if coef is not None:
    lag=row['model']['lag']
    for j in sorted({0,len(origins)-1}):
     o=int(origins[j]);other=companion(coef,u[o-128:o],y[o-128:o],u[o:o+128],lag);e=float(abs(other-pred[j]).max());scaled=e/max(1.,float(abs(other).max()));assert scaled<2e-7,(case,lag,j,e);max_companion_difference=max(max_companion_difference,e);checks.append(dict(case=case,lag=lag,origin=o,max_abs=e,scaled=scaled))
   elif row['method']=='persistence':assert np.array_equal(pred,np.asarray([np.full(128,y[o-1]) for o in origins]))
   else:assert row['method']=='train_mean' and np.count_nonzero(pred)==0
  error=np.concatenate(errs)
  assert abs(row['normalized_rmse']-np.sqrt(np.mean(error**2)))<1e-12 and abs(row['normalized_mae']-np.mean(abs(error)))<1e-12
 q=dict(utc=utc(),status='PASS_INDEPENDENT_COMPANION_ROLLOUT_AND_ALL_DEV_METRICS',canonical_result_sha256=sha(OUT/'CANONICAL_ARX74_V2.json'),data_manifest_sha256=sha(OUT/'TRAIN_DATA74_V2.json'),companion_checks=checks,max_companion_abs=max_companion_difference,max_saved_target_difference=max_target_difference,forecast_value_rows_rechecked=all_point_rows,per_system_method_rows=len(r['rows']),official_test_numeric_values_parsed=False,accepted_final_directions=0,local_result_files_written=0,wall_seconds=time.perf_counter()-started)
 dump(OUT/'CANONICAL_VERIFICATION74_V1.json',q);print(json.dumps({k:q[k] for k in ['status','max_companion_abs','max_saved_target_difference','forecast_value_rows_rechecked','per_system_method_rows','wall_seconds']}),flush=True)
if __name__=='__main__':main()
