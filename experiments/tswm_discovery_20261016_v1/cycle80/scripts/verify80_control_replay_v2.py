"""Compare exact ordinary-control replay after adding retained terminal capture."""
import sys,pathlib
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from native80_v1 import *
import numpy as np
a=read(OUT/'ORDINARY_RESULTS80_V1.json');b=read(OUT/'ORDINARY_RESULTS80_V2.json');v=read(OUT/'VERIFICATION80_V1.json');assert v['control_result_sha256']==sha(OUT/'ORDINARY_RESULTS80_V1.json') and read(OUT/'ordinary80_v2_exit.json')['exit_code']==0
assert len(a['rows'])==len(b['rows'])==12 and len(a['candidates'])==len(b['candidates'])==36
arraymax=0.;modelmax=0.
for left,right in zip(a['rows'],b['rows']):
 for key in ['system','initial_condition_index','role','method','privileged','finite_complete','mse','rmse','mae','nrmse','selected_library']:assert left[key]==right[key],key
 assert sha(left['array_path'])==left['array_sha256'] and sha(right['array_path'])==right['array_sha256']
 with np.load(left['array_path'],allow_pickle=False) as x,np.load(right['array_path'],allow_pickle=False) as y:
  assert set(x.files)==set(y.files)
  for key in x.files:
   assert np.array_equal(x[key],y[key]),key
   arraymax=max(arraymax,float(abs(x[key]-y[key]).max()))
for left,right in zip(a['candidates'],b['candidates']):
 for key in ['system','library','ridge','threshold','model','validation']:assert left[key]==right[key],key
for left,right in zip(a['models'],b['models']):
 assert sha(left['path'])==left['sha256'] and sha(right['path'])==right['sha256'];x=read(left['path']);y=read(right['path']);assert x==y
dump(OUT/'CONTROL_REPLAY_VERIFICATION80_V2.json',dict(utc=utc(),status='PASS_EXACT_LOGGED_ORDINARY_REPLAY_NO_SELECTION_OR_TUNING_CHANGE',original_result_sha256=sha(OUT/'ORDINARY_RESULTS80_V1.json'),logged_replay_result_sha256=sha(OUT/'ORDINARY_RESULTS80_V2.json'),original_independent_verification_sha256=sha(OUT/'VERIFICATION80_V1.json'),rows=12,context_selection_candidates=36,full_context_refits=3,array_max_abs_error=arraymax,all_model_coefficients_and_validation_metrics_exact=True,replay_is_not_independent_replication=True,reason='First structured artifacts retained, initial stdout only tool history; exact replay now retains server raw terminal log'))
print(json.dumps(dict(status='PASS_EXACT_LOGGED_REPLAY',rows=12,array_max_abs_error=arraymax)),flush=True)
