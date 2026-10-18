"""Independent scalar interval transport, support, target and contrast reconstruction."""
import sys,pathlib,time,ast
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common77_v1 import *
sys.path.insert(0,str(PROJECT/'cycle71'/'scripts'))
from runtime_affinity71_v1 import configure
AFFINITY=configure('efficient')
import numpy as np
def g(y,c):
 if c=='identity':return y.copy()
 if c=='affine':return 1.4*y+.35
 return 1.2*y+.25+.8*np.tanh(1.5*y)
def gi(x,c):
 if c=='identity':return x.copy()
 if c=='affine':return (x-.35)/1.4
 z=(x-.25)/1.2
 for _ in range(40):z-=(g(z,c)-x)/(1.2+1.2*(1-np.tanh(1.5*z)**2))
 assert float(abs(g(z,c)-x).max())<1e-10
 return z
def inverse(x,m):
 k=np.asarray(m['knots']);b=np.asarray(m['beta']);y=np.full_like(x,b[0],dtype=np.float64)
 for j in range(len(k)-1):
  q=np.minimum(np.maximum(x-k[j],0),k[j+1]-k[j])
  if j==0:q+=np.minimum(x-k[0],0)
  if j==len(k)-2:q+=np.maximum(x-k[-1],0)
  y+=b[j+1]*q
 return y
def forward(y,m):
 k=np.asarray(m['knots']);b=np.asarray(m['beta']);v=np.r_[b[0],b[0]+np.cumsum(np.diff(k)*b[1:])];i=np.clip(np.searchsorted(v,y,side='right')-1,0,len(k)-2)
 return k[i]+(y-v[i])/b[i+1]
def scalar(c,uh,yh,fu):
 u=list(uh);y=list(yh);out=[]
 for v in fu:
  z=c[0]+c[1]*v
  for lag in range(1,65):z+=c[1+lag]*u[-lag]+c[65+lag]*y[-lag]
  u.append(v);y.append(z);out.append(z)
 return np.asarray(out)
def checkscore(pred,target,expected):
 e=pred-target;finite=bool(np.isfinite(e).all());assert finite==expected['finite'] and e.size==expected['points']
 if not finite:return 0.
 mse=float(np.mean(e**2));rmse=float(np.sqrt(mse));mae=float(abs(e).mean());delta=max(abs(mse-expected['mse']),abs(rmse-expected['rmse']),abs(mae-expected['mae']));assert delta<1e-10;return delta
def main():
 started=time.perf_counter();r=read(OUT/'COMPOSITION_RESULTS77_V2.json');pin=read(OUT/'SOURCE_PIN77_V2.json');s75=read(PROJECT/'cycle75'/'outputs'/'DATA_SPLITS75_V1.json');m74=read(PROJECT/'cycle74'/'outputs'/'TRAIN_DATA74_V2.json');assert r['protocol_sha256']==sha(ROOT/'PROTOCOL77_V1.md') and r['source_pin_sha256']==sha(OUT/'SOURCE_PIN77_V2.json');checks=[];points=0;metricdiff=0.;canonicaldiff=0.;raw={};coefs={};diagonal=[]
 # Pin the actual imported source library against the previously published immutable snapshot.
 publication76=read(PROJECT/'cycle76'/'outputs'/'publication_v1'/'manifest.json');lib=next(x for x in publication76['files'] if x['path']=='scripts/simulation_calibration76_v1.py');assert sha(PROJECT/'cycle76'/'scripts'/'simulation_calibration76_v1.py')==lib['sha256']
 tree=ast.parse((ROOT/'scripts'/'run77_v2.py').read_text(encoding='utf-8-sig'));calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)];assert not any(name.startswith('fit') or name in ['minimize','lsq_linear','backward','step'] for name in calls)
 for name,cp in s75['source_models'].items():assert sha(cp['path'])==cp['sha256'];coefs[name]=np.load(cp['path'],allow_pickle=False)
 for a in pin['map_pins']:assert sha(a['map_path'])==a['map_sha256'] and sha(a['previous_diagonal_array'])==a['previous_diagonal_array_sha256']
 for rr in m74['records']:
  assert sha(rr['array_path'])==rr['array_sha256'];a=np.load(rr['array_path'],allow_pickle=False);sc=s75['scaling'][rr['dataset']];raw[rr['case']]=((a['u']-sc['mean'][0])/sc['std'][0],(a['y']-sc['mean'][1])/sc['std'][1])
 for row in r['rows']:
  assert sha(row['array_path'])==row['array_sha256'];a=np.load(row['array_path'],allow_pickle=False);p=a['pred'];target=a['target'];cp=a['canonical_pred'];ct=a['canonical_target'];ids=a['record_ids'];origins=a['origins'];inside=a['history_inside'];points+=p.size;c=row['condition'];dest=row['destination'];coef=coefs[dest];model=row['model'];hx=[]
  for i,(case,o) in enumerate(zip(ids,origins)):
   u,y=raw[str(case)];assert np.array_equal(ct[i],y[o:o+128]) and np.array_equal(target[i],g(y[o:o+128],c));hx.append(g(y[o-128:o],c))
  hx=np.asarray(hx);converted=gi(p,c);delta=float(abs(converted-cp).max());canonicaldiff=max(canonicaldiff,delta);assert delta<1e-10
  expected_inside=np.ones(len(ids),dtype=bool) if model is None else np.all((hx>=model['knots'][0])&(hx<=model['knots'][-1]),axis=1);assert np.array_equal(expected_inside,inside)
  if model is not None:
   outside_fraction=float(np.mean((hx<model['knots'][0])|(hx>model['knots'][-1])));assert outside_fraction==row['history_outside_value_fraction']
  metricdiff=max(metricdiff,checkscore(p,target,row['observed']),checkscore(cp,ct,row['canonical']));assert abs(row['observed']['rmse']/row['normalizer_eval_only_destination_calibration_x_std']-row['observed_nrmse'])<1e-12
  for label,mask in [('history_inside',inside),('history_outside',~inside)]:
   ss=next(z for z in row['support_strata'] if z['stratum']==label);assert int(mask.sum())==ss['origins']
   if mask.any():metricdiff=max(metricdiff,checkscore(p[mask],target[mask],ss['observed']),checkscore(cp[mask],ct[mask],ss['canonical']))
   else:assert ss['observed'] is None and ss['canonical'] is None
  for rr in row['per_record']:
   ix=ids==rr['case'];assert int(ix.sum())==rr['origins'];metricdiff=max(metricdiff,checkscore(p[ix],target[ix],rr['observed']),checkscore(cp[ix],ct[ix],rr['canonical']));assert int((ix&inside).sum())==rr['history_inside_origins'] and int((ix&~inside).sum())==rr['history_outside_origins']
   for i in np.flatnonzero(ix)[[0,-1]]:
    case=str(ids[i]);o=int(origins[i]);u,y=raw[case];uh=u[o-128:o];xh=hx[i];fu=u[o:o+128]
    if row['method']=='unadapted':q=scalar(coef,uh,xh,fu)
    elif row['method']=='known_sensor':q=g(scalar(coef,uh,y[o-128:o],fu),c)
    else:q=forward(scalar(coef,uh,inverse(xh,model),fu),model)
    delta=float(abs(q-p[i]).max());assert delta<1e-8*max(1.,float(abs(p[i]).max()));checks.append(dict(destination=dest,calibration_source=row['calibration_source'],condition=c,method=row['method'],case=case,origin=o,max_abs=delta))
  if row['calibration_source']==dest:
   prior=next(z for z in pin['map_pins'] if z['calibration_source']==dest and z['condition']==c and z['family']==row['family']);previous=np.load(prior['previous_diagonal_array'],allow_pickle=False);assert np.array_equal(previous['record_ids'],ids) and np.array_equal(previous['origins'],origins);delta=float(abs(p-previous['pred']).max());assert delta<1e-8;diagonal.append(dict(destination=dest,condition=c,family=row['family'],max_abs=delta))
 def mse(t,c,s=None,f=None,method=None):return next(z['canonical']['mse'] for z in r['rows'] if z['destination']==t and z['condition']==c and (z['method']==method if method else z['calibration_source']==s and z['family']==f))
 contrastdiff=0.
 for contrast in r['contrasts']:
  t=contrast['destination'];c=contrast['condition'];s=contrast['calibration_source'];f=contrast['family'];excess=mse(t,c,s,f)-mse(t,c,method='known_sensor');base=mse(t,'identity',s,f)-mse(t,'identity',method='known_sensor');change=excess-base;local=(mse(t,c,t,f)-mse(t,c,method='known_sensor'))-(mse(t,'identity',t,f)-mse(t,'identity',method='known_sensor'));delta=max(abs(excess-contrast['canonical_mse_excess_vs_known_sensor']),abs(base-contrast['identity_excess']),abs(change-contrast['condition_change_beyond_identity']),abs(change-local-contrast['change_minus_destination_diagonal']));assert delta<1e-12;contrastdiff=max(contrastdiff,delta)
 gridchecks=[]
 for grid in r['grid_diagnostics']:
  mp=next(z for z in pin['map_pins'] if z['calibration_source']==grid['calibration_source'] and z['condition']==grid['condition'] and z['family']==grid['family']);model=read(mp['map_path']);y=np.linspace(-2,2,1001);x=g(y,grid['condition']);mapped=inverse(x,model);covered=(x>=model['knots'][0])&(x<=model['knots'][-1]);err=mapped-y;delta=max(abs(float(np.sqrt(np.mean(err**2)))-grid['inverse_canonical_rmse']),abs(float(err.mean())-grid['inverse_canonical_bias']));assert delta<1e-10 and float(covered.mean())==grid['covered_grid_fraction'];gridchecks.append(dict(calibration_source=grid['calibration_source'],condition=grid['condition'],family=grid['family'],max_metric_abs=delta))
 result=dict(utc=utc(),status='FROZEN_TRANSPORT_ALL_TARGETS_METRICS_SUPPORT_CONTRASTS_AND_REPLAYS_VERIFIED',result_sha256=sha(OUT/'COMPOSITION_RESULTS77_V2.json'),forecast_values_verified=points,replay_checks=len(checks),max_replay_abs=max(z['max_abs'] for z in checks),metric_reconstruction_max_abs=metricdiff,canonical_inverse_max_abs=canonicaldiff,contrast_reconstruction_max_abs=contrastdiff,diagonal_checks=diagonal,grid_checks=gridchecks,checks=checks,no_new_fit_calls_in_runner=True,imported_library_publication_sha256=lib['sha256'],official_test_values_parsed=False,wall_seconds=time.perf_counter()-started)
 dump(OUT/'VERIFICATION77_V1.json',result);print(json.dumps({k:val for k,val in result.items() if k not in ['checks','diagonal_checks','grid_checks']}),flush=True)
if __name__=='__main__':main()
