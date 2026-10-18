"""Post-screen oracle path/sensitivity attribution; no model selection."""
import sys,pathlib,time
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common83_v1 import *
import run83_v1 as run
import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial import cKDTree
def jacobian(y,family,setting):
 if family=='hill':
  alpha,beta=[(2.,1.),(4.,.8)][setting];j=-beta*np.eye(7)
  for i in range(7):p=(i-1)%7;j[i,p]=-3*alpha*(1+.05*np.sin(i))*y[p]**2/(1+y[p]**3)**2
  return j
 g=[.15,.4][setting];j=np.zeros((12,12))
 for i in range(6):v=2*i;w=v+1;p=2*((i-1)%6);j[v,v]=1-y[v]**2-g;j[v,w]=-1.;j[v,p]=g;j[w,v]=.15+.02*i;j[w,w]=-.8*(.15+.02*i)
 return j
def nearest(x,q,parents):
 out=[]
 for idx in parents:
  xx=x[...,idx].reshape(-1,len(idx));m=xx.mean(0);s=np.maximum(xx.std(0),1e-8);out.append(cKDTree((xx-m)/s).query((q[:,idx]-m)/s)[0])
 return np.stack(out,-1)
def main():
 start=time.perf_counter();r=read(OUT/'RESULTS83_V1.json');assert read(OUT/'VERIFICATION83_V1.json')['status'].startswith('PASS_');manifest=read(OUT/'DATA_MANIFEST83_V1.json');assets=run.old.assets();w=next(z for z in assets if z['relative'].endswith('.safetensors'));model=run.load_fim_ode_local(pathlib.Path(w['path']).parent,'cpu');model.eval();rows=[];jac_checks=[];art=storage('artifact')/'attribution_v1';art.mkdir(exist_ok=False)
 for item in manifest['rows']:
  tag=item['tag'];assert sha(item['path'])==item['sha256']
  with np.load(item['path'],allow_pickle=False) as a:x=a['context'].copy();t=a['time'].copy()
  vf=run.Native(model,x,t,item['parents'])
  for row in [z for z in r['rollouts'] if z['tag']==tag and z['kind']=='FIM_local_composed']:
   if not row['finite_complete'] or not row['solver']['success']:rows.append(dict(tag=tag,initial_index=row['initial_index'],clamp=row['clamp'],status='INCOMPLETE_NATIVE_ROLLOUT_PRESERVED'));continue
   with np.load(row['path'],allow_pickle=False) as a:y=a['target'].copy();p=a['prediction'].copy();grid=a['time'].copy()
   true=run.data.true_field(y,item['family'],item['setting']);truep=run.data.true_field(p,item['family'],item['setting']);est=vf(y);estp=vf(p);defect=est-true;defectp=estp-truep;js=np.stack([jacobian(yy,item['family'],item['setting']) for yy in y]);clamp=row['clamp'];den=np.maximum(true.std(0),1e-12);dist=nearest(x,y,item['parents']);distp=nearest(x,p,item['parents']);indices=list(range(1,len(den))) if clamp else list(range(len(den)))
   check_y=y[0];eps=1e-5;num=np.column_stack([(run.data.true_field(check_y+np.eye(len(den))[i]*eps,item['family'],item['setting'])-run.data.true_field(check_y-np.eye(len(den))[i]*eps,item['family'],item['setting']))/(2*eps) for i in range(len(den))]);je=float(abs(num-jacobian(check_y,item['family'],item['setting'])).max());assert je<1e-7;jac_checks.append(je)
   if clamp:js[:,0,:]=0.;defect[:,0]=0.;defectp[:,0]=0.
   def rhs(now,delta):
    k=min(max(int(np.searchsorted(grid,now))-1,0),len(grid)-2);weight=(now-grid[k])/(grid[k+1]-grid[k]);j=(1-weight)*js[k]+weight*js[k+1];d=(1-weight)*defect[k]+weight*defect[k+1];return j@delta+d
   sol=solve_ivp(rhs,(grid[0],grid[-1]),np.zeros(len(den)),t_eval=grid,method='DOP853',rtol=1e-7,atol=1e-9);assert sol.success;delta=sol.y.T;actual=p-y;norm=np.linalg.norm(actual[:,indices]);surrogate_error=float(np.linalg.norm((delta-actual)[:,indices])/max(norm,1e-12));cos=float(np.sum(delta[:,indices]*actual[:,indices])/(max(np.linalg.norm(delta[:,indices])*norm,1e-12)));bins=[]
   for name,mask in [('near_0_.1',dist<=.1),('mid_.1_.5',(dist>.1)&(dist<=.5)),('far_above_.5',dist>.5)]:
    mask=mask[:,indices];value=(defect/den)[:,indices];bins.append(dict(bin=name,coordinates=int(mask.sum()),normalized_defect_rmse=float(np.sqrt(np.mean(value[mask]**2))) if mask.any() else None))
   path=art/f"{tag}_ic{row['initial_index']}_clamp{int(clamp)}.npz";np.savez_compressed(path,time=grid,reference_field=true,estimated_reference_field=est,reference_defect=defect,prediction_defect=defectp,reference_distance=dist,prediction_distance=distp,linearized_error=delta,actual_error=actual)
   out=dict(tag=tag,family=item['family'],noise_sigma=item['noise_sigma'],initial_index=row['initial_index'],clamp=clamp,status='COMPLETE_ORACLE_DIAGNOSTIC',reference_field_defect=float(np.mean(np.sqrt(np.mean(defect[:,indices]**2,0))/den[indices])),prediction_field_defect=float(np.mean(np.sqrt(np.mean(defectp[:,indices]**2,0))/den[indices])),reference_support_p95=float(np.quantile(dist[:,indices],.95)),prediction_support_p95=float(np.quantile(distp[:,indices],.95)),linearized_relative_error=surrogate_error,linearized_actual_cosine=cos,reference_support_bins=bins,path=str(path),sha256=sha(path));rows.append(out)
  print(json.dumps(dict(tag=tag,diagnostic_rows=len(rows))),flush=True)
 dump(OUT/'ATTRIBUTION83_V1.json',dict(utc=utc(),status='POST_SCREEN_ORACLE_ATTRIBUTION_NOT_A_METHOD',protocol_sha256=sha(ROOT/'ATTRIBUTION_PROTOCOL83_V1.md'),source_results_sha256=sha(OUT/'RESULTS83_V1.json'),rows=rows,analytic_jacobian_checks=len(jac_checks),jacobian_max_abs=max(jac_checks),wall_seconds=time.perf_counter()-start,unavailable_truth_used_for_attribution_only=True,accepted_final_directions=0));print(json.dumps(dict(status='ATTRIBUTION_CLOSED',rows=len(rows),jacobian_max_abs=max(jac_checks))),flush=True)
if __name__=='__main__':main()
