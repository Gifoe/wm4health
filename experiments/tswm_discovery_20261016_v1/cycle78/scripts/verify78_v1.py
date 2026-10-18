"""Independent historical sufficient statistics, constrained solves and scalar forecasts."""
import sys,pathlib,time,ast
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common78_v1 import *
sys.path.insert(0,str(PROJECT/'cycle71'/'scripts'))
from runtime_affinity71_v1 import configure
AFFINITY=configure('efficient')
import numpy as np
def g(y,c):
 if c=='identity':return y.copy()
 if c=='affine':return y*1.4+.35
 return y*1.2+.25+.8*np.tanh(y*1.5)
def gi(x,c):
 if c=='identity':return x.copy()
 if c=='affine':return (x-.35)/1.4
 y=(x-.25)/1.2
 for _ in range(40):y-=(g(y,c)-x)/(1.2+1.2*(1-np.tanh(1.5*y)**2))
 return y
def inverse(x,m):
 k=np.asarray(m['knots']);b=np.asarray(m['beta']);out=np.full_like(x,b[0],dtype=np.float64)
 for j in range(len(k)-1):
  d=np.minimum(np.maximum(x-k[j],0),k[j+1]-k[j])
  if j==0:d+=np.minimum(x-k[0],0)
  if j==len(k)-2:d+=np.maximum(x-k[-1],0)
  out+=b[j+1]*d
 return out
def forward(y,m):
 k=np.asarray(m['knots']);b=np.asarray(m['beta']);v=b[0]+np.r_[0,np.cumsum(np.diff(k)*b[1:])];ix=np.clip(np.searchsorted(v,y,side='right')-1,0,len(k)-2)
 return k[ix]+(y-v[ix])/b[ix+1],1/b[ix+1]
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
  t=64+j;value=np.full(len(u),coef[0])+coef[1]*u[:,t]
  for lag in range(1,65):value+=coef[1+lag]*u[:,t-lag]+coef[65+lag]*y[:,t-lag]
  y[:,t]=value
 return y[:,64:]
def boxsolve(A,b):
 # Complete active-set enumeration for a strictly convex2D ridge least-squares problem.
 candidates=[];lo=np.array([.02,-2.]);hi=np.array([50.,2.]);unconstrained=np.linalg.lstsq(A,b,rcond=None)[0]
 if np.all(unconstrained>=lo) and np.all(unconstrained<=hi):candidates.append(unconstrained)
 for q in [.02,50.]:
  r=float(np.clip(A[:,1]@(b-q*A[:,0])/(A[:,1]@A[:,1]),-2,2));candidates.append(np.array([q,r]))
 for r in [-2.,2.]:
  q=float(np.clip(A[:,0]@(b-r*A[:,1])/(A[:,0]@A[:,0]),.02,50));candidates.append(np.array([q,r]))
 return min(candidates,key=lambda x:float(np.sum((A@x-b)**2)))
def checkscore(p,y,s):
 e=p-y;finite=bool(np.isfinite(e).all());assert finite==s['finite'] and e.size==s['points']
 if not finite:return 0.
 mse=float(np.mean(e**2));rmse=float(np.sqrt(mse));mae=float(abs(e).mean());delta=max(abs(mse-s['mse']),abs(rmse-s['rmse']),abs(mae-s['mae']));assert delta<1e-10;return delta
def main():
 started=time.perf_counter();result=read(OUT/'ALIGNMENT_RESULTS78_V1.json');pin=read(OUT/'SOURCE_PIN78_V1.json');s75=read(PROJECT/'cycle75'/'outputs'/'DATA_SPLITS75_V1.json');m74=read(PROJECT/'cycle74'/'outputs'/'TRAIN_DATA74_V2.json');raw={};coefs={};checks=[];points=0;metricdiff=0.;paramdiff=0.;costdiff=0.;parameterchecks=0;pastchecks=0;pastgradient=[];canonicaldiff=0.
 assert result['protocol_sha256']==sha(ROOT/'PROTOCOL78_V1.md') and result['source_pin_sha256']==sha(OUT/'SOURCE_PIN78_V1.json')
 helper=ast.parse((ROOT/'scripts'/'history_alignment78_v1.py').read_text());func=next(x for x in helper.body if isinstance(x,ast.FunctionDef) and x.name=='fit_history');assert [a.arg for a in func.args.args]==['coef','m','uh','xh','residual_scale','method'];assert not any(isinstance(x,(ast.Import,ast.ImportFrom)) for x in ast.walk(func))
 for name,cp in pin['source_models'].items():assert sha(cp['path'])==cp['sha256'];coefs[name]=np.load(cp['path'],allow_pickle=False)
 for rr in m74['records']:
  assert sha(rr['array_path'])==rr['array_sha256'];a=np.load(rr['array_path'],allow_pickle=False);sc=s75['scaling'][rr['dataset']];raw[rr['case']]=((a['u']-sc['mean'][0])/sc['std'][0],(a['y']-sc['mean'][1])/sc['std'][1])
 noisechecks=[]
 for name,co in coefs.items():
  squares=0.;n=0
  for split in s75['splits']:
   if split['dataset']!=name:continue
   u,y=raw[split['case']];t=np.arange(64,split['source'][1]);p=np.full(len(t),co[0])+co[1]*u[t]
   for lag in range(1,65):p+=co[lag+1]*u[t-lag]+co[lag+65]*y[t-lag]
   squares+=float(np.sum((p-y[t])**2));n+=len(t)
  expected=max(squares/n,1e-12);delta=abs(expected-pin['source_residual_scales'][name]['innovation_ridge']);assert delta<1e-12;noisechecks.append(dict(dataset=name,source_rows=n,max_abs=delta))
 for row in result['rows']:
  assert sha(row['array_path'])==row['array_sha256'] and sha(row['fit_metadata_path'])==row['fit_metadata_sha256'];a=np.load(row['array_path'],allow_pickle=False);meta=read(row['fit_metadata_path']);assert meta['parameters_sha256']==row['array_sha256'];p=a['pred'];target=a['target'];cp=a['canonical_pred'];ct=a['canonical_target'];par=a['parameters'];ids=a['record_ids'];origins=a['origins'];points+=p.size;dataset=row['destination'];condition=row['condition'];coef=coefs[dataset];mp=next(z for z in pin['map_pins'] if z['calibration_source']==row['calibration_source'] and z['condition']==condition and z['family']==row['family']);assert sha(mp['map_path'])==mp['map_sha256'];model=read(mp['map_path']);uh=[];xh=[]
  assert np.isfinite(par).all() and np.all(par[:,0]>=.02) and np.all(par[:,0]<=50) and np.all(par[:,1]>=-2) and np.all(par[:,1]<=2)
  for i,(case,o) in enumerate(zip(ids,origins)):
   u,y=raw[str(case)];assert np.array_equal(ct[i],y[o:o+128]) and np.array_equal(target[i],g(y[o:o+128],condition));uh.append(u[o-128:o]);xh.append(g(y[o-128:o],condition));assert int(meta['origins'][i])==o and meta['record_ids'][i]==str(case)
  uh=np.asarray(uh);xh=np.asarray(xh);z=inverse(xh,model);converted=gi(p,condition);delta=float(abs(cp-converted).max());canonicaldiff=max(canonicaldiff,delta);assert delta<1e-10;metricdiff=max(metricdiff,checkscore(p,target,row['observed']),checkscore(cp,ct,row['canonical']));assert abs(row['observed']['rmse']/row['evaluation_only_x_std']-row['observed_nrmse'])<1e-12
  if row['method'] in ['offset_innovation_ridge','affine_innovation_ridge']:
   lam=pin['source_residual_scales'][dataset]['innovation_ridge'];D=float(1-coef[66:].sum());V=z[:,64:].copy();driver=np.full_like(V,coef[0])+coef[1]*uh[:,64:]
   for lag in range(1,65):V-=coef[65+lag]*z[:,64-lag:128-lag];driver+=coef[1+lag]*uh[:,64-lag:128-lag]
   for i in range(len(par)):
    if row['method']=='offset_innovation_ridge':
     aa=np.r_[np.full(64,-D),np.sqrt(lam)][:,None];bb=np.r_[V[i]-driver[i],0.];ref=np.array([1.,float(np.clip(np.linalg.lstsq(aa,bb,rcond=None)[0][0],-2,2))]);cost=float(np.sum((aa[:,0]*ref[1]-bb)**2))
    else:
     A=np.concatenate([np.column_stack([driver[i],np.full(64,-D)]),np.sqrt(lam)*np.eye(2)]);b=np.r_[V[i],np.sqrt(lam),0.];ref=boxsolve(A,b);cost=float(np.sum((A@par[i]-b)**2))
    delta=float(abs(ref-par[i]).max());paramdiff=max(paramdiff,delta);assert delta<1e-6;cd=abs(cost-meta['metadata'][i]['cost']);costdiff=max(costdiff,cd);assert cd<1e-8;parameterchecks+=1
  else:
   # Independent direct recursion of the fitted map on the observed past64 outputs.
   q=par[:,0,None];r=par[:,1,None];latent=batch(coef,uh[:,:64],(z[:,:64]+r)/q,uh[:,64:]);past,slope=forward(q*latent-r,model);scale=np.maximum(xh.std(axis=1),.001);res=(past-xh[:,64:])/scale[:,None];cost=np.sum(res**2,axis=1)+.001*np.sum((par-np.array([1.,0.]))**2,axis=1);cd=float(abs(cost-np.asarray([m['cost'] for m in meta['metadata']])).max());costdiff=max(costdiff,cd);assert cd<1e-8
   homogeneous=coef.copy();homogeneous[:66]=0;d=batch(coef,uh[:,:64],np.zeros_like(z[:,:64]),uh[:,64:]);s=batch(homogeneous,np.zeros_like(uh[:,:64]),np.ones_like(z[:,:64]),np.zeros_like(uh[:,64:]))-1;J=np.stack([slope*d/scale[:,None],slope*s/scale[:,None]],axis=-1);gradient=np.einsum('nhi,nh->ni',J,res)+.001*(par-np.array([1.,0.]));projected=gradient.copy();projected[(par<=np.array([.02,-2.])+1e-8)&(gradient>0)]=0;projected[(par>=np.array([50.,2.])-1e-8)&(gradient<0)]=0;pastgradient.append(dict(destination=dataset,source=row['calibration_source'],condition=condition,family=row['family'],origins=len(par),solver_nonconverged=sum(not m['success'] for m in meta['metadata']),max_projected_gradient_abs=float(abs(projected).max()),past_cost_reconstruction_max_abs=cd));pastchecks+=len(par)
  for rr in row['per_record']:
   mask=ids==rr['case'];metricdiff=max(metricdiff,checkscore(p[mask],target[mask],rr['observed']),checkscore(cp[mask],ct[mask],rr['canonical']));assert rr['nuisance']['origins']==int(mask.sum())
   for i in np.flatnonzero(mask)[[0,-1]]:
    case=str(ids[i]);o=int(origins[i]);u,y=raw[case];q,r=par[i];history=(inverse(xh[i],model)+r)/q;latent=scalar(coef,uh[i],history,u[o:o+128]);prediction,_=forward(q*latent-r,model);delta=float(abs(prediction-p[i]).max());assert delta<1e-8*max(1.,float(abs(p[i]).max()));checks.append(dict(destination=dataset,calibration_source=row['calibration_source'],condition=condition,family=row['family'],method=row['method'],case=case,origin=o,max_abs=delta))
  assert row['nuisance']['solver_successes']==sum(x['success'] for x in meta['metadata']) and row['nuisance']['solver_nonconverged']==sum(not x['success'] for x in meta['metadata'])
 verification=dict(utc=utc(),status='HISTORY_ONLY_SUFFICIENT_STATISTICS_CONSTRAINED_SOLVES_AND_FORECASTS_VERIFIED',result_sha256=sha(OUT/'ALIGNMENT_RESULTS78_V1.json'),forecast_values_verified=points,scalar_replay_checks=len(checks),scalar_replay_max_abs=max(z['max_abs'] for z in checks),metric_reconstruction_max_abs=metricdiff,canonical_inverse_reconstruction_max_abs=canonicaldiff,innovation_solutions_independently_checked=parameterchecks,innovation_parameter_max_abs=paramdiff,history_cost_max_abs=costdiff,past_simulation_costs_checked=pastchecks,source_residual_checks=noisechecks,independent_past_gradient_diagnostics=pastgradient,checks=checks,fit_api_has_no_future_or_generator_arguments=True,official_test_values_parsed=False,wall_seconds=time.perf_counter()-started)
 dump(OUT/'VERIFICATION78_V1.json',verification);print(json.dumps({k:val for k,val in verification.items() if k not in ['checks','independent_past_gradient_diagnostics']}),flush=True)
if __name__=='__main__':main()
