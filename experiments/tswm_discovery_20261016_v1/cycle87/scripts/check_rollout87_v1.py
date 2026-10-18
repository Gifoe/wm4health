from native87_v1 import *
from evaluate87_v2 import NativeField
from empirical87_v1 import load_prior,weak_design,posterior
from ordinary87_v1 import predict as polynomial,FAMILIES
from scipy.integrate import solve_ivp
import numpy as np,time,gc
def main():
 started=time.perf_counter();result=read(O/'EVALUATION_RESULTS87_V3.json');selection=read(O/'SELECTION87_V1.json');emp=read(O/'EMPIRICAL_SELECTION87_V1.json');specs=[('native_frozen',None)]+[(f"native_seed{f['seed']}_{f['arm']}",f['selected']['checkpoint_path']) for f in selection['fits']]+[('empirical_prior_'+e['arm'],e) for e in emp['selections']];rows=[];m=load_native();initial={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
 for name,package in specs:
  native=name.startswith('native')
  if native:
   if package:
    checkpoint=torch.load(package,map_location='cpu',weights_only=False);m.load_state_dict(checkpoint['state_dict']);del checkpoint
   else:m.load_state_dict(initial)
   m.eval()
   for parameter in m.parameters():parameter.requires_grad_(False)
  for cohort in ['retained','crossed','unselected']:
   data=[]
   for fam in FAMILIES:
    with np.load(RAW/'evaluation_v2'/f'{fam}_{cohort}_v1.npz') as z:data.append({k:z[k] for k in z.files})
   row=next(r for r in result['rows'] if r['model']==name and r['cohort']==cohort)
   with np.load(row['array_path']) as z:stored=z['prediction']
   indices=[]
   for fi in range(2):
    amplitude=np.nanmax(abs(stored[fi*128:(fi+1)*128]),axis=(1,2));indices.extend([fi*128,fi*128+int(np.nanargmax(amplitude))])
   indices=np.unique(indices);context=np.concatenate([d['context'] for d in data])[indices];query=np.concatenate([d['query_initial'] for d in data])[indices]
   if native:
    fn=NativeField(m,context)
    def rhs(t,x):
     with torch.no_grad():return fn(torch.as_tensor(x.reshape(len(indices),2),device='cuda',dtype=torch.float32)).cpu().numpy().astype(np.float64).ravel()
    rtol=1e-5;atol=1e-7
   else:
    h,y=weak_design(context);coeff=posterior(load_prior(package['prior_path']),h,y,package['weak_noise_std'])
    def rhs(t,x):return polynomial(coeff,x.reshape(len(indices),2),3).ravel()
    rtol=2e-10;atol=2e-11
   sol=solve_ivp(rhs,(0,10),query.ravel(),method='DOP853',t_eval=np.arange(201)*.05,rtol=rtol,atol=atol,max_step=.1);assert sol.success and sol.y.shape==(len(indices)*2,201),(name,cohort,sol.message);ref=sol.y.T.reshape(201,len(indices),2).transpose(1,0,2);expected=stored[indices];assert np.isfinite(expected).all()
   absmax=float(abs(ref-expected).max());scaled=float((np.sqrt(np.mean((ref-expected)**2,axis=(1,2)))/np.maximum(np.sqrt(np.mean(ref*ref,axis=(1,2))),.1)).max());row=dict(model=name,cohort=cohort,indices=indices.tolist(),nfev=int(sol.nfev),max_abs=absmax,max_trajectory_rms_relative=scaled,rtol=rtol,atol=atol);rows.append(row);log(dict(stage='independent_learned_rollout_check',**row));assert scaled<.01,(name,cohort,absmax,scaled)
 write(O/'ROLLOUT_NUMERICS87_V1.json',dict(utc=utc(),status='PASS_INDEPENDENT_DOP853_FIXED_AND_WORST_AMPLITUDE_SUBSET',rows=rows,total_case_trajectories=sum(len(r['indices']) for r in rows),max_trajectory_rms_relative=max(r['max_trajectory_rms_relative'] for r in rows),max_abs=max(r['max_abs'] for r in rows),full_population_independent_reintegration=False,original_results_sha256=sha(O/'EVALUATION_RESULTS87_V2.json'),revised_results_sha256=sha(O/'EVALUATION_RESULTS87_V3.json'),script_sha256=sha(__file__),seconds=time.perf_counter()-started));log(dict(stage='independent_rollout_numerics_complete',rows=len(rows),seconds=time.perf_counter()-started))
if __name__=='__main__':main()
