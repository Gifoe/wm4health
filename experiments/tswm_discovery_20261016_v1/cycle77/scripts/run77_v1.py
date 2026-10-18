import sys,pathlib,time
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common77_v1 import *
sys.path.insert(0,str(PROJECT/'cycle71'/'scripts'))
from runtime_affinity71_v1 import configure
AFFINITY=configure('efficient')
import numpy as np
sys.path.insert(0,str(PROJECT/'cycle76'/'scripts'))
import simulation_calibration76_v1 as sim
def g(y,c):
 if c=='identity':return y.copy()
 if c=='affine':return .35+1.4*y
 return .25+1.2*y+.8*np.tanh(1.5*y)
def gi(x,c):
 if c=='identity':return x.copy()
 if c=='affine':return (x-.35)/1.4
 lo=(x-.25-.8)/1.2-1.;hi=(x-.25+.8)/1.2+1.
 for _ in range(60):
  mid=(lo+hi)/2;upper=g(mid,c)>x;hi=np.where(upper,mid,hi);lo=np.where(upper,lo,mid)
 return (lo+hi)/2
def batch_arx(coef,uh,yh,fu):
 n,H=fu.shape;u=np.concatenate([uh[:,-64:],fu],axis=1);y=np.concatenate([yh[:,-64:].copy(),np.zeros((n,H))],axis=1)
 for j in range(H):
  t=64+j;ix=t-np.arange(1,65);y[:,t]=coef[0]+coef[1]*u[:,t]+u[:,ix]@coef[2:66]+y[:,ix]@coef[66:]
 return y[:,64:]
def transport(coef,model,uh,xh,fu):
 # No generator, destination calibration outputs or future targets enter this API.
 k=np.asarray(model['knots']);beta=np.asarray(model['beta']);R,d=sim.pack(coef,k,uh,xh,fu);return sim.prediction(R,d,k,beta)
def score(p,target):
 e=p-target;finite=bool(np.isfinite(e).all())
 if finite:
  with np.errstate(over='ignore'):mse=float(np.mean(e**2));mae=float(np.mean(abs(e)))
  finite=bool(np.isfinite(mse) and np.isfinite(mae))
 return dict(finite=finite,points=int(e.size),mse=mse if finite else None,rmse=float(np.sqrt(mse)) if finite else None,mae=mae if finite else None)
def main():
 started=time.perf_counter();art=storage('artifact')/'composition_v1';art.mkdir(exist_ok=False);s75=read(PROJECT/'cycle75'/'outputs'/'DATA_SPLITS75_V1.json');r75=read(PROJECT/'cycle75'/'outputs'/'CALIBRATION_RESULTS75_V1.json');r76=read(PROJECT/'cycle76'/'outputs'/'SIMULATION_RESULTS76_V1.json');m74=read(PROJECT/'cycle74'/'outputs'/'TRAIN_DATA74_V2.json');coef={};raw={};maps={};pins=[];rows=[];gridrows=[];diagonal=[]
 for dataset,cp in s75['source_models'].items():assert sha(cp['path'])==cp['sha256'];coef[dataset]=np.load(cp['path'],allow_pickle=False)
 for r in m74['records']:
  assert sha(r['array_path'])==r['array_sha256'];a=np.load(r['array_path'],allow_pickle=False);sc=s75['scaling'][r['dataset']];raw[r['case']]=((a['u']-sc['mean'][0])/sc['std'][0],(a['y']-sc['mean'][1])/sc['std'][1])
 families=['C75_innovation_monotone','C76_simulation_affine','C76_simulation_monotone']
 for source in ['EMPS','F16']:
  for condition in ['identity','affine','nonlinear']:
   for family in families:
    if family.startswith('C75'):
     item=next(r for r in r75['rows'] if r['dataset']==source and r['condition']==condition and r['method']=='source_monotone_inverse');path=item['model']['path'];expected=item['model']['sha256'];previous_array=item['arrays_path'];previous_array_sha=item['arrays_sha256']
    else:
     method='simulation_affine_global' if family=='C76_simulation_affine' else 'simulation_monotone_selected';item=next(r for r in r76['rows'] if r['dataset']==source and r['condition']==condition and r['method']==method);path=item['map_path'];expected=item['map_sha256'];previous_array=item['array_path'];previous_array_sha=item['array_sha256']
    assert sha(path)==expected;model=read(path);maps[source,condition,family]=model;pins.append(dict(calibration_source=source,condition=condition,family=family,map_path=path,map_sha256=expected,previous_diagonal_array=previous_array,previous_diagonal_array_sha256=previous_array_sha))
    canonical=np.linspace(-2,2,1001);x=g(canonical,condition);mapped=sim.inverse_map(x,np.asarray(model['knots']),np.asarray(model['beta']));support=(x>=model['knots'][0])&(x<=model['knots'][-1]);gridrows.append(dict(calibration_source=source,condition=condition,family=family,canonical_grid=[-2.,2.,1001],inverse_canonical_rmse=float(np.sqrt(np.mean((mapped-canonical)**2))),inverse_canonical_bias=float(np.mean(mapped-canonical)),covered_grid_fraction=float(support.mean()),covered_grid_rmse=float(np.sqrt(np.mean((mapped[support]-canonical[support])**2))) if support.any() else None,canonical_physical_identification_claim=False))
 dump(OUT/'SOURCE_PIN77_V1.json',dict(utc=utc(),protocol_sha256=sha(ROOT/'PROTOCOL77_V1.md'),split_sha256=sha(PROJECT/'cycle75'/'outputs'/'DATA_SPLITS75_V1.json'),results75_sha256=sha(PROJECT/'cycle75'/'outputs'/'CALIBRATION_RESULTS75_V1.json'),results76_sha256=sha(PROJECT/'cycle76'/'outputs'/'SIMULATION_RESULTS76_V1.json'),source_models=s75['source_models'],map_pins=pins,raw_arrays_remain_server_only=True,official_test_values_parsed=False))
 for target in ['EMPS','F16']:
  records=[r for r in m74['records'] if r['dataset']==target]
  for condition in ['identity','affine','nonlinear']:
   uh=[];yh=[];fu=[];canonical=[];ids=[];origins=[]
   for r in records:
    u,y=raw[r['case']]
    for o in r['dev_origins']:uh.append(u[o-128:o]);yh.append(y[o-128:o]);fu.append(u[o:o+128]);canonical.append(y[o:o+128]);ids.append(r['case']);origins.append(o)
   uh,yh,fu,canonical=map(np.asarray,[uh,yh,fu,canonical]);ids=np.asarray(ids);origins=np.asarray(origins);xh=g(yh,condition);target_values=g(canonical,condition);denom=next(r['normalizer_calibration_x_std'] for r in r75['rows'] if r['dataset']==target and r['condition']==condition)
   def close(method,source,family,p,model=None):
    cp=gi(p,condition);observed=score(p,target_values);cs=score(cp,canonical);byrecord=[]
    if model is not None:
     k=model['knots'];out=(xh<k[0])|(xh>k[-1]);inside=~out.any(axis=1);outfraction=float(out.mean())
    else:inside=np.ones(len(ids),dtype=bool);outfraction=None
    support=[]
    for label,mask in [('history_inside',inside),('history_outside',~inside)]:
     support.append(dict(stratum=label,origins=int(mask.sum()),observed=score(p[mask],target_values[mask]) if mask.any() else None,canonical=score(cp[mask],canonical[mask]) if mask.any() else None))
    for case in dict.fromkeys(ids.tolist()):
     mask=ids==case;os=score(p[mask],target_values[mask]);byrecord.append(dict(case=case,origins=int(mask.sum()),observed=os,observed_nrmse=os['rmse']/denom if os['finite'] else None,canonical=score(cp[mask],canonical[mask]),history_inside_origins=int((inside&mask).sum()),history_outside_origins=int((~inside&mask).sum())))
    dest=art/(target+'_'+condition+'_'+method+'.npz');np.savez_compressed(dest,pred=p,target=target_values,canonical_pred=cp,canonical_target=canonical,record_ids=ids,origins=origins,history_inside=inside)
    row=dict(destination=target,condition=condition,method=method,calibration_source=source,family=family,cross_system=source is not None and source!=target,destination_new_view_calibration_available_to_arm=source==target if source else False,privileged=method=='known_sensor',observed=observed,observed_nrmse=observed['rmse']/denom if observed['finite'] else None,normalizer_eval_only_destination_calibration_x_std=denom,canonical=cs,per_record=byrecord,support_strata=support,history_outside_value_fraction=outfraction,model=model,array_path=str(dest),array_sha256=sha(dest));rows.append(row);print(json.dumps({key:row[key] for key in ['destination','condition','method','observed_nrmse','history_outside_value_fraction']}),flush=True)
    if source==target:
     pin=next(v for v in pins if v['calibration_source']==source and v['condition']==condition and v['family']==family);assert sha(pin['previous_diagonal_array'])==pin['previous_diagonal_array_sha256'];previous=np.load(pin['previous_diagonal_array'],allow_pickle=False)
     if family.startswith('C75'):
      ordered=np.concatenate([previous[r['case']+'_pred'] for r in records]);delta=float(abs(ordered-p).max())
     else:assert np.array_equal(previous['record_ids'],ids) and np.array_equal(previous['origins'],origins);delta=float(abs(previous['pred']-p).max())
     assert delta<1e-8;diagonal.append(dict(destination=target,condition=condition,family=family,max_prediction_abs_difference=delta))
   close('unadapted',None,None,batch_arx(coef[target],uh,xh,fu))
   close('known_sensor',None,None,g(batch_arx(coef[target],uh,yh,fu),condition))
   for source in ['EMPS','F16']:
    for family in families:close(source+'_'+family,source,family,transport(coef[target],maps[source,condition,family],uh,xh,fu),maps[source,condition,family])
   state('PROGRESS77_V1.json',dict(status='DESTINATION_CONDITION_CLOSED',destination=target,condition=condition,closed_rows=len(rows)))
 contrasts=[]
 def cmse(t,c,source=None,family=None,method=None):return next(v['canonical']['mse'] for v in rows if v['destination']==t and v['condition']==c and (v['method']==method if method else v['calibration_source']==source and v['family']==family))
 for target in ['EMPS','F16']:
  for source in ['EMPS','F16']:
   for family in families:
    identity_excess=cmse(target,'identity',source,family)-cmse(target,'identity',method='known_sensor');diagonal_identity=cmse(target,'identity',target,family)-cmse(target,'identity',method='known_sensor')
    for condition in ['identity','affine','nonlinear']:
     excess=cmse(target,condition,source,family)-cmse(target,condition,method='known_sensor');diagonal_excess=cmse(target,condition,target,family)-cmse(target,condition,method='known_sensor');change=excess-identity_excess;interaction=change-(diagonal_excess-diagonal_identity);contrasts.append(dict(destination=target,calibration_source=source,condition=condition,family=family,cross_system=source!=target,canonical_mse_excess_vs_known_sensor=excess,identity_excess=identity_excess,condition_change_beyond_identity=change,change_minus_destination_diagonal=interaction))
 result=dict(utc=utc(),status='FROZEN_COMPONENT_CROSS_COMPOSITION_DEV_MATRIX_CLOSED',protocol_sha256=sha(ROOT/'PROTOCOL77_V1.md'),source_pin_sha256=sha(OUT/'SOURCE_PIN77_V1.json'),rows=rows,contrasts=contrasts,grid_diagnostics=gridrows,diagonal_previous_prediction_checks=diagonal,new_fits=0,shared_operator_in_normalized_coordinates_not_physical_units=True,retrospective_not_untouched_confirmation=True,official_test_values_parsed=False,new_method=False,accepted_final_directions=0,overall_goal='ACTIVE_UNACHIEVED',process_affinity=AFFINITY,wall_seconds=time.perf_counter()-started,local_result_files_written=0)
 dump(OUT/'COMPOSITION_RESULTS77_V1.json',result);state('PROGRESS77_V1.json',dict(status=result['status'],rows=len(rows),wall_seconds=result['wall_seconds']));print(json.dumps(dict(status=result['status'],rows=len(rows),wall_seconds=result['wall_seconds'])),flush=True)
if __name__=='__main__':main()
