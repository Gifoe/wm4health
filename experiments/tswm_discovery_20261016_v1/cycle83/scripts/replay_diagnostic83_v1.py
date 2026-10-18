"""Quantify a failed exact-equality check without changing numerical outcomes."""
import sys,pathlib,time
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common83_v1 import *
import run83_v1 as run
import numpy as np
def main():
 start=time.perf_counter();r=read(OUT/'RESULTS83_V1.json');m=read(OUT/'DATA_MANIFEST83_V1.json');asset=next(z for z in run.old.assets() if z['relative'].endswith('.safetensors'));model=run.load_fim_ode_local(pathlib.Path(asset['path']).parent,'cpu');model.eval();rows=[]
 for freeze in [False,True]:
  for parameter in model.parameters():parameter.requires_grad_(not freeze)
  for item in m['rows']:
   with np.load(item['path'],allow_pickle=False) as a:x=a['context'].copy();t=a['time'].copy();q=a['queries'].copy()
   row=next(z for z in r['fields'] if z['tag']==item['tag'] and z['kind']=='FIM_local_composed')
   with np.load(row['path'],allow_pickle=False) as a:stored=a['prediction'].copy();truth=a['target'].copy()
   fn=run.Native(model,x,t,item['parents']);pred=fn(q);again=fn(q);err=float(abs(pred-stored).max());repeat=float(abs(pred-again).max());ma=run.metrics(pred,truth);mb=run.metrics(stored,truth);rows.append(dict(tag=item['tag'],parameters_frozen=freeze,stored_replay_max_abs=err,repeated_query_max_abs=repeat,exact_match=bool(np.array_equal(pred,stored)),within_original_interface_tolerance=bool(np.allclose(pred,stored,atol=2e-5,rtol=2e-5)),field_score_difference=abs(ma['coordinate_mean_nrmse']-mb['coordinate_mean_nrmse'])));
 assert run.old.check_source()==70 and sha(asset['path'])==asset['sha256'];dump(OUT/'NATIVE_REPLAY_DIAGNOSTIC83_V1.json',dict(utc=utc(),status='EXACT_CHECK_FAILURE_QUANTIFIED_NOT_A_MODEL_CHANGE',rows=rows,original_interface_atol=2e-5,original_interface_rtol=2e-5,source_and_weights_unchanged=True,wall_seconds=time.perf_counter()-start));print(json.dumps(dict(rows=rows)),flush=True)
if __name__=='__main__':main()
