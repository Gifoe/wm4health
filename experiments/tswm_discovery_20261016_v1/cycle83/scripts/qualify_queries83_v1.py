"""Check that field diagnostics and one-location solver calls denote the same field."""
import sys,pathlib,time
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common83_v1 import *
import run83_v1 as run
import numpy as np
def main():
 start=time.perf_counter();m=read(OUT/'DATA_MANIFEST83_V1.json');asset=next(z for z in run.old.assets() if z['relative'].endswith('.safetensors'));model=run.load_fim_ode_local(pathlib.Path(asset['path']).parent,'cpu');model.eval();rows=[]
 for family in ['fhn','hill']:
  for sigma in [0.,.03]:
   item=next(z for z in m['rows'] if z['family']==family and z['noise_sigma']==sigma)
   with np.load(item['path'],allow_pickle=False) as a:x=a['context'].copy();t=a['time'].copy();q=a['queries'].copy()
   fn=run.Native(model,x,t,item['parents']);allq=fn(q);short=fn(q[:17]);single=np.concatenate([fn(z[None]) for z in q[:17]],0);reversedq=fn(q[::-1])[::-1];duplicate=fn(np.r_[q,q[:17]])[:128]
   for name,pred,reference in [('prefix17_vs_all128',short,allq[:17]),('single1_vs17',single,short),('reverse128',reversedq,allq),('duplicate145',duplicate,allq)]:
    err=float(abs(pred-reference).max());assert np.allclose(pred,reference,atol=2e-5,rtol=2e-5),(item['tag'],name,err);rows.append(dict(tag=item['tag'],check=name,max_abs=err,scalar_count=pred.size))
 assert run.old.check_source()==70 and sha(asset['path'])==asset['sha256'];dump(OUT/'QUERY_QUALIFICATION83_V1.json',dict(utc=utc(),status='PASS_QUERY_SET_INDEPENDENCE_WITH_FIXED_TOLERANCE',checks=rows,atol=2e-5,rtol=2e-5,wall_seconds=time.perf_counter()-start));print(json.dumps(dict(status='PASS',checks=len(rows),maximum_absolute_error=max(z['max_abs'] for z in rows))),flush=True)
if __name__=='__main__':main()
