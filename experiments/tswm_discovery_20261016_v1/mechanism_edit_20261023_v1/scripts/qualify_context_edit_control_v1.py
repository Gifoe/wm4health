import pathlib,json,sys,hashlib,datetime
import numpy as np,jax.numpy as jnp
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from context_edit_controls_v1 import persistent_context_edit
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'editor_contract_recovery_v1';rows=[]
for B in [1,2,10]:
 c=jnp.broadcast_to(jnp.arange(8,dtype=jnp.float32)[None,None,:]/8,(B,64,8))
 for delta in [.5,-.25,jnp.arange(1,B+1,dtype=jnp.float32)/4]:
  p=persistent_context_edit(c,6,delta);d=np.asarray(p-c);expected=np.asarray(delta) if np.ndim(delta) else np.repeat(float(delta),B)
  assert np.array_equal(d[:,:,6],np.broadcast_to(expected[:,None],(B,64))) and np.count_nonzero(d[:,:,:6])==0 and np.count_nonzero(d[:,:,7:])==0
  rows.append(dict(batch=B,shift=expected.tolist(),time_constant=True,only_dimension6_changed=True))
with (OUT/'ORDINARY_EDIT_CONTROL_V1.json').open('x',encoding='utf-8') as f:json.dump(dict(status='ORDINARY_FIXED_AXIS_SCALE_AND_TIME_CONTROL_QUALIFIED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cases=rows,source_sha256=hashlib.sha256((ROOT/'scripts/context_edit_controls_v1.py').read_bytes()).hexdigest(),full_recorder_or_clock_alignment_repaired=False,learned_model_executed=False,new_method=False),f,indent=2)
print(json.dumps(dict(status='ORDINARY_EDIT_CONTROL_QUALIFIED',cases=len(rows),new_method=False)),flush=True)
