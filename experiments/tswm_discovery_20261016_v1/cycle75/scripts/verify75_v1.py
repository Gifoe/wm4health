import sys,pathlib,time
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common75_v1 import *
sys.path.insert(0,str(PROJECT/'cycle71'/'scripts'))
from runtime_affinity71_v1 import configure
AFFINITY=configure('efficient')
import numpy as np
import torch
torch.set_num_threads(2)
def g(y,c):
 if c=='identity':return y.copy()
 if c=='affine':return y*1.4+.35
 return y*1.2+.25+.8*np.tanh(y*1.5)
def gi(x,c):
 if c=='identity':return x.copy()
 if c=='affine':return (x-.35)/1.4
 y=(x-.25)/1.2
 for _ in range(40):y-=(g(y,c)-x)/(1.2+1.2*(1-np.tanh(1.5*y)**2))
 assert abs(g(y,c)-x).max()<1e-10
 return y
def inverse(x,m):
 k=np.asarray(m['knots']);b=np.asarray(m['beta']);y=b[0]+np.zeros_like(x,dtype=np.float64)
 for j in range(len(k)-1):
  d=np.minimum(np.maximum(x-k[j],0),k[j+1]-k[j])
  if j==0:d+=np.minimum(x-k[0],0)
  if j==len(k)-2:d+=np.maximum(x-k[-1],0)
  y+=b[j+1]*d
 return y
def forward(y,m):
 k=np.asarray(m['knots']);b=np.asarray(m['beta']);v=np.r_[b[0],b[0]+np.cumsum(np.diff(k)*b[1:])];i=np.searchsorted(v,y,side='right')-1;i=np.clip(i,0,len(k)-2)
 return k[i]+(y-v[i])/b[i+1]
def recurrence(coef,uh,yh,fu):
 up=list(np.asarray(uh));yp=list(np.asarray(yh));out=[]
 for current in fu:
  y=coef[0]+coef[1]*current
  for j in range(1,65):y+=coef[1+j]*up[-j]+coef[65+j]*yp[-j]
  yp.append(y);up.append(current);out.append(y)
 return np.asarray(out)
def neural(obj,uh,xh,fu):
 weights={k:v.numpy() for k,v in obj['state_dict'].items()};mu=obj['observed_mean'];sd=obj['observed_std'];up=list(np.asarray(uh,np.float32));yp=list(np.asarray((xh-mu)/sd,np.float32));out=[]
 for current in fu:
  z=np.r_[np.float32(current),np.asarray(up[-64:][::-1]),np.asarray(yp[-64:][::-1])].astype(np.float32)
  for layer in [0,2]:
   z=weights[str(layer)+'.weight']@z+weights[str(layer)+'.bias'];z=z*np.exp(-np.logaddexp(np.float32(0),-z))
  increment=(weights['4.weight']@z+weights['4.bias'])[0];y=np.float32(yp[-1]+increment);yp.append(y);up.append(np.float32(current));out.append(float(y)*sd+mu)
 return np.asarray(out)
def main():
 started=time.perf_counter();result=read(OUT/'CALIBRATION_RESULTS75_V1.json');splits=read(OUT/'DATA_SPLITS75_V1.json');base=read(PROJECT/'cycle74'/'outputs'/'TRAIN_DATA74_V2.json');assert result['protocol_sha256']==sha(ROOT/'PROTOCOL75_V1.md');assert result['data_splits_sha256']==sha(OUT/'DATA_SPLITS75_V1.json');raw={};source={};checks=[];points=0;maxmetric=0.;max_target=0.;mode=[]
 for dataset,model in splits['source_models'].items():
  assert sha(model['path'])==model['sha256'];coef=np.load(model['path'],allow_pickle=False);source[dataset]=coef;comp=np.zeros((64,64));comp[0,:]=coef[66:];comp[1:,:-1]=np.eye(63);mode.append(dict(dataset=dataset,ar_spectral_radius=float(abs(np.linalg.eigvals(comp)).max()),offset_innovation_multiplier=float(1-coef[66:].sum())))
 for r in base['records']:
  assert sha(r['array_path'])==r['array_sha256'];a=np.load(r['array_path'],allow_pickle=False);s=splits['scaling'][r['dataset']];raw[r['case']]=((a['u']-s['mean'][0])/s['std'][0],(a['y']-s['mean'][1])/s['std'][1])
 for row in result['rows']:
  assert sha(row['arrays_path'])==row['arrays_sha256'];a=np.load(row['arrays_path'],allow_pickle=False);pred=a['pred'];target=a['target'];ids=a['record_ids'];origins=a['origins'];points+=pred.size;condition=row['condition'];method=row['method'];dataset=row['dataset'];info=row['model'];coef=source[dataset];nn=None
  if method=='calibration_arx64':assert sha(info['path'])==info['sha256'];coef=np.load(info['path'],allow_pickle=False)
  if method.startswith('calibration_narx') and 'artifact' in info:
   assert sha(info['artifact']['path'])==info['artifact']['sha256'];nn=torch.load(info['artifact']['path'],map_location='cpu',weights_only=True)
  for i,(case,o) in enumerate(zip(ids,origins)):
   u,y=raw[str(case)];expected=g(y[o:o+128],condition);diff=float(abs(expected-target[i]).max());max_target=max(max_target,diff);assert diff==0
  for rr in row['per_record']:
   ix=ids==rr['case'];e=pred[ix]-target[ix];finite=bool(np.isfinite(e).all());assert finite==rr['finite']
   if finite:
    rmse=float(np.sqrt(np.mean(e**2)));mae=float(abs(e).mean());diff=max(abs(rmse-rr['rmse']),abs(mae-rr['mae']),abs(rmse/row['normalizer_calibration_x_std']-rr['nrmse']));maxmetric=max(maxmetric,diff);assert diff<1e-12
   chosen=np.flatnonzero(ix)[[0,-1]]
   for i in chosen:
    case=str(ids[i]);o=int(origins[i]);u,y=raw[case];uh=u[o-128:o];xh=g(y[o-128:o],condition);fu=u[o:o+128]
    if not rr['finite']:continue
    if method=='persistence':p=np.full(128,xh[-1])
    elif method=='source_unadapted' or method=='calibration_arx64':p=recurrence(coef,uh,xh,fu)
    elif method=='privileged_known_sensor':p=g(recurrence(coef,uh,gi(xh,condition),fu),condition)
    elif method in ['source_affine_inverse','source_monotone_inverse']:
     m=info['adapter'];assert sha(info['path'])==info['sha256'];p=forward(recurrence(coef,uh,inverse(xh,m),fu),m)
    else:assert nn is not None;p=neural(nn,uh,xh,fu)
    delta=abs(p-pred[i]);maxdiff=float(delta.max());tol=(.002 if nn is not None else 1e-8)*max(1.,float(abs(pred[i]).max()));assert maxdiff<=tol,(dataset,condition,method,case,int(o),maxdiff,tol)
    checks.append(dict(dataset=dataset,condition=condition,method=method,case=case,origin=o,max_abs=maxdiff,first_step_abs=float(delta[0]),tolerance=tol))
  if row['all_records_finite']:
   e=pred-target;rmse=float(np.sqrt(np.mean(e**2)));diff=max(abs(rmse-row['rmse']),abs(rmse/row['normalizer_calibration_x_std']-row['nrmse']),abs(float(abs(e).mean())-row['mae']));maxmetric=max(maxmetric,diff);assert diff<1e-12
 selection=[]
 for item in result['adapter_selection']:
  cs=[c for c in item['candidates'] if c['internal_validation_nrmse'] is not None];best=min(cs,key=lambda c:c['internal_validation_nrmse']);assert best['lambda_identity_slopes']==item['selected_lambda'];m=read(item['path']);coef=source[item['dataset']];r=next(s for s in splits['splits'] if s['dataset']==item['dataset']);u,y=raw[r['case']];start,end=r['calibration_fit'];x=g(y[start:end],item['condition']);uf=u[start:end];h=inverse(x,m);errors=[]
  # Direct innovation versus separately formed basis features, selected representative TRAIN-calibration points.
  k=np.asarray(m['knots']);beta=np.asarray(m['beta'])
  def explicit_basis(v):
   z=[1.]
   for j in range(len(k)-1):
    q=min(max(v-k[j],0),k[j+1]-k[j])
    if j==0:q+=min(v-k[0],0)
    if j==len(k)-2:q+=max(v-k[-1],0)
    z.append(q)
   return np.asarray(z)
  for t in [64,65,len(x)-1]:
   direct=h[t]-sum(coef[65+j]*h[t-j] for j in range(1,65));d=explicit_basis(x[t])-sum((coef[65+j]*explicit_basis(x[t-j]) for j in range(1,65)),start=np.zeros(len(beta)));errors.append(abs(float(d@beta)-direct))
  assert max(errors)<1e-9;selection.append(dict(dataset=item['dataset'],condition=item['condition'],kind=item['kind'],selection_uses_internal_validation_only=True,innovation_design_max_abs=float(max(errors))))
 summary=dict(utc=utc(),status='ALL_STORED_METRICS_TARGETS_AND_INDEPENDENT_REPLAYS_VERIFIED',result_sha256=sha(OUT/'CALIBRATION_RESULTS75_V1.json'),verification_protocol_sha256=sha(ROOT/'VERIFICATION_PROTOCOL75_V1.md'),forecast_values_verified=points,max_target_difference=max_target,max_metric_reconstruction_difference=maxmetric,replay_checks=len(checks),float64_max_replay_abs=max(c['max_abs'] for c in checks if not c['method'].startswith('calibration_narx')),float32_neural_max_replay_abs=max(c['max_abs'] for c in checks if c['method'].startswith('calibration_narx')),source_model_diagnostics=mode,adapter_selection_checks=selection,checks=checks,official_test_values_parsed=False,process_affinity=AFFINITY,wall_seconds=time.perf_counter()-started)
 dump(OUT/'VERIFICATION75_V1.json',summary);print(json.dumps({k:v for k,v in summary.items() if k not in ['checks','adapter_selection_checks','process_affinity']}),flush=True)
if __name__=='__main__':main()
