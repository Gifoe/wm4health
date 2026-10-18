from native87_v1 import *
from evaluate87_v2 import NativeField
from scipy.integrate import solve_ivp
import numpy as np,time
class NativeDoubleField:
 def __init__(self,m,context):
  self.m=m;b,t,n,_=context.shape;tt=np.broadcast_to((np.arange(n)*.05)[None,None,:,None],(b,t,n,1)).copy()
  with torch.no_grad():self.enc,self.mask,self.concept=m.trajectory_encoding(torch.as_tensor(pad(context),device='cuda',dtype=torch.float64),torch.as_tensor(tt,device='cuda',dtype=torch.float64),torch.ones((b,t,n,1),device='cuda',dtype=torch.bool))
 def __call__(self,x):
  q=torch.nn.functional.pad(x[:,None,:],(0,1));q=self.m.spatial_norm.normalization_map(q,self.concept._states_norm_stats)
  with torch.no_grad():r=self.m.function_decoding(q,self.mask,self.enc,self.concept);return self.m.get_prediction_for_eval(r)[:,0,:2]
def main():
 result=read(O/'EVALUATION_RESULTS87_V3.json');r=next(r for r in result['rows'] if r['model']=='native_frozen' and r['cohort']=='retained');ds=[]
 for fam in ['duffing','stuart_landau']:
  with np.load(RAW/'evaluation_v2'/f'{fam}_retained_v1.npz') as z:ds.append({k:z[k] for k in z.files})
 with np.load(r['array_path']) as z:stored=z['prediction']
 ix=np.unique([0,int(np.max(abs(stored[:128]),axis=(1,2)).argmax()),128,128+int(np.max(abs(stored[128:]),axis=(1,2)).argmax())]);context=np.concatenate([d['context'] for d in ds])[ix];initial=np.concatenate([d['query_initial'] for d in ds])[ix];m=load_native();m.eval()
 for p in m.parameters():p.requires_grad_(False)
 fn=NativeField(m,context);x=torch.as_tensor(initial,device='cuda',dtype=torch.float32);a=fn(x).cpu().numpy();b=fn(x).cpu().numpy();det=float(abs(a-b).max());del fn;m.double();fn=NativeDoubleField(m,context);v=fn(torch.as_tensor(initial,device='cuda',dtype=torch.float64)).cpu().numpy();log(dict(stage='fp64_probe_opened',indices=ix.tolist(),float32_repeat_max_abs=det,float32_vs_float64_field_max_abs=float(abs(a-v).max())))
 calls=0;started=time.perf_counter()
 def rhs(t,x):
  nonlocal calls
  calls+=1
  if calls%1000==0:log(dict(stage='fp64_probe_rhs',calls=calls,t=float(t),seconds=time.perf_counter()-started))
  if calls>10000:raise RuntimeError('Probe function-evaluation budget exceeded; preserve diagnostic, no restart of existing job')
  with torch.no_grad():return fn(torch.as_tensor(x.reshape(len(ix),2),device='cuda',dtype=torch.float64)).cpu().numpy().ravel()
 sol=solve_ivp(rhs,(0,10),initial.ravel(),t_eval=np.arange(201)*.05,method='DOP853',rtol=1e-7,atol=1e-9,max_step=.1);assert sol.success;ref=sol.y.T.reshape(201,len(ix),2).transpose(1,0,2);delta=float(abs(ref-stored[ix]).max());rms=float((np.sqrt(np.mean((ref-stored[ix])**2,axis=(1,2)))/np.maximum(np.sqrt(np.mean(ref**2,axis=(1,2))),.1)).max());receipt=dict(utc=utc(),status='FP64_REFERENCE_PROBE_CLOSED',indices=ix.tolist(),calls=calls,seconds=time.perf_counter()-started,float32_repeat_max_abs=det,float32_vs_float64_field_max_abs=float(abs(a-v).max()),max_abs_delta_vs_float32_rk4=delta,max_trajectory_rms_relative=rms,weights_changed=False,precision_cast_only=True);write(O/'FP64_PROBE87_V1.json',receipt);log(dict(stage='fp64_probe_closed',**{k:v for k,v in receipt.items() if k!='utc'}))
if __name__=='__main__':main()
