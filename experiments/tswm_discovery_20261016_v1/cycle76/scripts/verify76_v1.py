"""Independent scalar/batched recurrence, affine optimum and complete score reconstruction."""
import sys,pathlib,time
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common76_v1 import *
sys.path.insert(0,str(PROJECT/'cycle71'/'scripts'))
from runtime_affinity71_v1 import configure
AFFINITY=configure('efficient')
import numpy as np
def g(y,c):
 if c=='identity':return y.copy()
 if c=='affine':return 1.4*y+.35
 return 1.2*y+.25+.8*np.tanh(1.5*y)
def inverse(x,m):
 k=np.asarray(m['knots']);b=np.asarray(m['beta']);out=np.full_like(x,b[0],dtype=np.float64)
 for j in range(len(k)-1):
  v=np.minimum(np.maximum(x-k[j],0),k[j+1]-k[j])
  if j==0:v+=np.minimum(x-k[0],0)
  if j==len(k)-2:v+=np.maximum(x-k[-1],0)
  out+=b[j+1]*v
 return out
def forward(y,m):
 k=np.asarray(m['knots']);b=np.asarray(m['beta']);v=b[0]+np.r_[0,np.cumsum(b[1:]*np.diff(k))];out=np.empty_like(y)
 for i,value in enumerate(np.asarray(y).ravel()):
  j=max(0,min(len(k)-2,int(np.searchsorted(v,value,side='right'))-1));out.ravel()[i]=k[j]+(value-v[j])/b[j+1]
 return out
def scalar(coef,uh,yh,fu):
 u=list(uh);y=list(yh);out=[]
 for v in fu:
  z=coef[0]+coef[1]*v
  for j in range(1,65):z+=coef[j+1]*u[-j]+coef[j+65]*y[-j]
  u.append(v);y.append(z);out.append(z)
 return np.asarray(out)
def batch(coef,uh,yh,fu):
 u=np.concatenate([uh[:,-64:],fu],axis=1);y=np.concatenate([yh[:,-64:].copy(),np.zeros_like(fu)],axis=1)
 for j in range(fu.shape[1]):
  t=j+64;pred=np.full(len(u),coef[0])+coef[1]*u[:,t]
  for lag in range(1,65):pred+=coef[lag+1]*u[:,t-lag]+coef[lag+65]*y[:,t-lag]
  y[:,t]=pred
 return y[:,64:]
def main():
 started=time.perf_counter();r=read(OUT/'SIMULATION_RESULTS76_V1.json');s=read(PROJECT/'cycle75'/'outputs'/'DATA_SPLITS75_V1.json');m=read(PROJECT/'cycle74'/'outputs'/'TRAIN_DATA74_V2.json');assert r['protocol_sha256']==sha(ROOT/'PROTOCOL76_V1.md');raw={};coeff={};checks=[];optima=[];points=0;maxmetric=0.;selection=[]
 for name,cp in s['source_models'].items():assert sha(cp['path'])==cp['sha256'];coeff[name]=np.load(cp['path'],allow_pickle=False)
 for rr in m['records']:
  assert sha(rr['array_path'])==rr['array_sha256'];a=np.load(rr['array_path'],allow_pickle=False);sc=s['scaling'][rr['dataset']];raw[rr['case']]=((a['u']-sc['mean'][0])/sc['std'][0],(a['y']-sc['mean'][1])/sc['std'][1])
 for row in r['rows']:
  assert sha(row['array_path'])==row['array_sha256'] and sha(row['map_path'])==row['map_sha256'];a=np.load(row['array_path'],allow_pickle=False);model=read(row['map_path']);p=a['pred'];y=a['target'];ids=a['record_ids'];origins=a['origins'];coef=coeff[row['dataset']];points+=p.size;condition=row['condition'];scale=row['normalizer_calibration_x_std']
  for i,(case,o) in enumerate(zip(ids,origins)):
   u,rawy=raw[str(case)];assert np.array_equal(y[i],g(rawy[o:o+128],condition))
  for rr in row['per_record']:
   ix=ids==rr['case'];e=p[ix]-y[ix];finite=bool(np.isfinite(e).all());assert finite==rr['finite']
   if finite:
    rmse=float(np.sqrt(np.mean(e**2)));mae=float(np.mean(abs(e)));delta=max(abs(rmse-rr['rmse']),abs(rmse/scale-rr['nrmse']),abs(mae-rr['mae']));assert delta<1e-12;maxmetric=max(maxmetric,delta)
   for i in np.flatnonzero(ix)[[0,-1]]:
    if not finite:continue
    case=str(ids[i]);o=int(origins[i]);u,rawy=raw[case];xh=g(rawy[o-128:o],condition);q=forward(scalar(coef,u[o-128:o],inverse(xh,model),u[o:o+128]),model);delta=float(abs(q-p[i]).max());assert delta<1e-8*max(1.,float(abs(p[i]).max()));checks.append(dict(dataset=row['dataset'],condition=condition,method=row['method'],case=case,origin=o,max_abs=delta))
  if row['all_records_finite']:
   e=p-y;rmse=float(np.sqrt(np.mean(e**2)));delta=max(abs(rmse-row['rmse']),abs(rmse/scale-row['nrmse']),abs(float(abs(e).mean())-row['mae']));assert delta<1e-12;maxmetric=max(maxmetric,delta)
 for fit in r['fits']:
  dataset=fit['dataset'];condition=fit['condition'];coef=coeff[dataset];homogeneous=coef.copy();homogeneous[:66]=0;uh=[];xh=[];fu=[];target=[];vh=[];vx=[];vf=[];vy=[]
  for contract in r['origin_contracts']:
   if contract['dataset']!=dataset or contract['condition']!=condition:continue
   u,y=raw[contract['case']];x=g(y,condition)
   for origins,columns in [(contract['fit_origins'],[uh,xh,fu,target]),(contract['internal_origins'],[vh,vx,vf,vy])]:
    for o in origins:
     for out,value in zip(columns,[u[o-128:o],x[o-128:o],u[o:o+128],x[o:o+128]]):out.append(value)
  uh,xh,fu,target,vh,vx,vf,vy=map(np.asarray,[uh,xh,fu,target,vh,vx,vf,vy]);zeros=np.zeros_like(xh);d=batch(coef,uh,zeros,fu);Rx=batch(homogeneous,np.zeros_like(uh),xh,np.zeros_like(fu));R1=batch(homogeneous,np.zeros_like(uh),np.ones_like(xh),np.zeros_like(fu));D=np.column_stack([d.ravel(),(R1-1).ravel()]);rhs=(target-Rx).ravel();unconstrained=np.linalg.lstsq(D,rhs,rcond=None)[0];q=float(np.clip(unconstrained[0],.02,50.));z=R1.ravel()-1;intercept=float(z@(rhs-q*d.ravel())/(z@z)) if q!=unconstrained[0] else float(unconstrained[1]);ref=np.array([q,intercept]);actual=np.array([fit['affine']['q'],fit['affine']['r']]);delta=float(abs(ref-actual).max());assert delta<1e-8;objective=float(np.mean((Rx+q*d+intercept*(R1-1)-target)**2));assert abs(objective-fit['affine']['training_simulation_mse'])<1e-12;optima.append(dict(dataset=dataset,condition=condition,global_affine_independent_lstsq_max_abs=delta,training_mse_reconstruction_abs=abs(objective-fit['affine']['training_simulation_mse']),fit_origins=len(target)))
  candidates=[(c['internal_validation_mse'],c) for attempt in fit['monotone_optimizer_attempts'] for c in attempt['checkpoints'] if c['internal_validation_mse'] is not None];best=min(candidates,key=lambda p:p[0]);selected=fit['monotone_selected'];assert best[0]==selected['internal_validation_normalized_mse'];assert best[1]['initializer']==selected['selected_initializer'] and best[1]['iteration']==selected['selected_iteration'] and best[1]['stage']==selected['selected_stage'];latent=batch(coef,vh,inverse(vx,selected),vf)
  # Vectorized independent interval inverse for the complete internal-validation replay.
  k=np.asarray(selected['knots']);b=np.asarray(selected['beta']);v=b[0]+np.r_[0,np.cumsum(b[1:]*np.diff(k))];idx=np.clip(np.searchsorted(v,latent,side='right')-1,0,len(k)-2);forecast=k[idx]+(latent-v[idx])/b[idx+1];score=float(np.mean(((forecast-vy)/fit['normalizer_calibration_x_std'])**2));assert abs(score-best[0])<1e-10;selection.append(dict(dataset=dataset,condition=condition,selected_initializer=selected['selected_initializer'],selected_iteration=selected['selected_iteration'],selected_stage=selected['selected_stage'],internal_score_reconstruction_abs=abs(score-best[0]),DEV_reselection=False))
 output=dict(utc=utc(),status='INDEPENDENT_GLOBAL_AFFINE_OPTIMA_REPLAYS_AND_COMPLETE_METRICS_VERIFIED',result_sha256=sha(OUT/'SIMULATION_RESULTS76_V1.json'),forecast_values_verified=points,replay_checks=len(checks),max_replay_abs=max(c['max_abs'] for c in checks),max_metric_reconstruction_abs=maxmetric,affine_global_optima_checks=optima,internal_selection_checks=selection,checks=checks,official_test_values_parsed=False,wall_seconds=time.perf_counter()-started)
 dump(OUT/'VERIFICATION76_V1.json',output);print(json.dumps({k:v for k,v in output.items() if k!='checks'}),flush=True)
if __name__=='__main__':main()
