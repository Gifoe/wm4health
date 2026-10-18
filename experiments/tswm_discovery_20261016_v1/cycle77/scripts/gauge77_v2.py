import sys,pathlib,time
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common77_v1 import *
sys.path.insert(0,str(PROJECT/'cycle71'/'scripts'))
# The independent verifier configures process affinity once on import.
import numpy as np
sys.path.insert(0,str(PROJECT/'cycle76'/'scripts'))
import simulation_calibration76_v1 as sim
import verify77_v1 as independent
AFFINITY=independent.AFFINITY
def main():
 started=time.perf_counter();result=read(OUT/'COMPOSITION_RESULTS77_V2.json');pins=read(OUT/'SOURCE_PIN77_V2.json');s75=read(PROJECT/'cycle75'/'outputs'/'DATA_SPLITS75_V1.json');m74=read(PROJECT/'cycle74'/'outputs'/'TRAIN_DATA74_V2.json');art=storage('artifact')/'offset_profiles_v2';art.mkdir(exist_ok=False);raw={};coefs={};data={};profiles=[];arrays=[];checks=[];response=[];formula_max=0.
 for name,cp in s75['source_models'].items():assert sha(cp['path'])==cp['sha256'];coefs[name]=np.load(cp['path'],allow_pickle=False)
 for rr in m74['records']:
  assert sha(rr['array_path'])==rr['array_sha256'];a=np.load(rr['array_path'],allow_pickle=False);sc=s75['scaling'][rr['dataset']];raw[rr['case']]=((a['u']-sc['mean'][0])/sc['std'][0],(a['y']-sc['mean'][1])/sc['std'][1])
 for dataset in ['EMPS','F16']:
  c=coefs[dataset].copy();homogeneous=c.copy();homogeneous[:66]=0.;r1=independent.scalar(homogeneous,np.zeros(128),np.ones(128),np.zeros(128));fixture=homogeneous.copy();fixture[66]=1-fixture[67:].sum();exact=independent.scalar(fixture,np.zeros(128),np.ones(128),np.zeros(128));assert abs(exact-1).max()<1e-12
  response.append(dict(dataset=dataset,one_minus_sum_output_lags=float(1-c[66:].sum()),homogeneous_R1_minus_one_rms=float(np.sqrt(np.mean((r1-1)**2))),homogeneous_R1_minus_one_max_abs=float(abs(r1-1).max()),R1_last=float(r1[-1]),unit_sum_mathematical_fixture_max_abs=float(abs(exact-1).max()),physical_nonidentifiability_proven=False))
  for condition in ['identity','affine','nonlinear']:
   for phase in ['internal','dev']:
    columns=[[],[],[],[]];ids=[];origins=[]
    for split in s75['splits']:
     if split['dataset']!=dataset:continue
     u,y=raw[split['case']];x=independent.g(y,condition);oo=split['internal_origins'] if phase=='internal' else split['dev_origins']
     for o in oo:
      for out,val in zip(columns,[u[o-128:o],x[o-128:o],u[o:o+128],x[o:o+128]]):out.append(val)
      ids.append(split['case']);origins.append(o)
    data[dataset,condition,phase]=(tuple(np.asarray(z) for z in columns),np.asarray(ids),np.asarray(origins))
 for pin in pins['map_pins']:
  source=pin['calibration_source'];condition=pin['condition'];family=pin['family'];assert sha(pin['map_path'])==pin['map_sha256'];model=read(pin['map_path']);k=np.asarray(model['knots']);beta=np.asarray(model['beta']);diagnostic=next(z for z in result['grid_diagnostics'] if z['calibration_source']==source and z['condition']==condition and z['family']==family);oracle=-diagnostic['inverse_canonical_bias'];offsets=[('fixed_'+str(delta),delta,False) for delta in [-1.,-.5,0.,.5,1.]]+[('privileged_grid_offset',oracle,True)]
  for destination,phase in [(source,'internal'),('EMPS','dev'),('F16','dev')]:
   batch,ids,origins=data[destination,condition,phase];uh,xh,fu,target=batch;R,d=sim.pack(coefs[destination],k,uh,xh,fu);base=R@beta+d;r1=R[...,0];baseline=sim.forward_map(base,k,beta);canonical_target=independent.gi(target,condition);pp=[]
   denom=next(z['normalizer_eval_only_destination_calibration_x_std'] for z in result['rows'] if z['destination']==destination and z['condition']==condition)
   for label,delta,privileged in offsets:
    shifted=beta.copy();shifted[0]+=delta;p=sim.forward_map(base+delta*(r1-1),k,beta);direct=sim.prediction(R,d,k,shifted);error=float(abs(p-direct).max());formula_max=max(formula_max,error);assert error<1e-10;cp=independent.gi(p,condition);observed_rmse=float(np.sqrt(np.mean((p-target)**2)));canonical_rmse=float(np.sqrt(np.mean((cp-canonical_target)**2)));change=float(np.sqrt(np.mean((p-baseline)**2)));profiles.append(dict(calibration_source=source,destination=destination,condition=condition,family=family,phase=phase,offset_label=label,delta=float(delta),privileged_offset=privileged,origins=len(ids),observed_rmse=observed_rmse,observed_nrmse=observed_rmse/denom,canonical_rmse=canonical_rmse,prediction_change_rms=change,prediction_change_normalized=change/denom));pp.append(p)
    # Independent recurrence and shifted interval functions, first/last origin of every record.
    changed=dict(model,beta=shifted.tolist())
    for case in dict.fromkeys(ids.tolist()):
     for i in np.flatnonzero(ids==case)[[0,-1]]:
      q=independent.forward(independent.scalar(coefs[destination],uh[i],independent.inverse(xh[i],changed),fu[i]),changed);diff=float(abs(q-p[i]).max());assert diff<1e-8*max(1.,float(abs(p[i]).max()));checks.append(diff)
   filename=art/(source+'_'+destination+'_'+condition+'_'+family+'_'+phase+'.npz');np.savez_compressed(filename,pred=np.asarray(pp),target=target,record_ids=ids,origins=origins,offsets=np.array([x[1] for x in offsets]),labels=np.array([x[0] for x in offsets]));arrays.append(dict(calibration_source=source,destination=destination,condition=condition,family=family,phase=phase,path=str(filename),sha256=sha(filename)))
 metricdiff=0.;points=0
 for file in arrays:
  assert sha(file['path'])==file['sha256'];a=np.load(file['path'],allow_pickle=False);points+=a['pred'].size
  for i,label in enumerate(a['labels']):
   row=next(z for z in profiles if all(z[k]==file[k] for k in ['calibration_source','destination','condition','family','phase']) and z['offset_label']==str(label));p=a['pred'][i];target=a['target'];rmse=float(np.sqrt(np.mean((p-target)**2)));canonical=float(np.sqrt(np.mean((independent.gi(p,file['condition'])-independent.gi(target,file['condition']))**2)));metricdiff=max(metricdiff,abs(rmse-row['observed_rmse']),abs(canonical-row['canonical_rmse']));assert metricdiff<1e-12
 output=dict(utc=utc(),status='POSTDISCOVERY_FIXED_OFFSET_PROFILES_AND_LINEAR_RESPONSE_IDENTITY_VERIFIED',protocol_sha256=sha(ROOT/'GAUGE_DIAGNOSTIC77_PROTOCOL_V1.md'),composition_result_sha256=sha(OUT/'COMPOSITION_RESULTS77_V2.json'),profiles=profiles,artifacts=arrays,source_response=response,profile_rows=len(profiles),forecast_values_verified=points,independent_scalar_replay_checks=len(checks),independent_scalar_replay_max_abs=max(checks),translation_formula_max_abs=formula_max,metric_reconstruction_max_abs=metricdiff,original_maps_changed=False,offset_selected_from_DEV=False,privileged_offset_used_as_method=False,new_fits=0,accepted_final_directions=0,overall_goal='ACTIVE_UNACHIEVED',official_test_values_parsed=False,wall_seconds=time.perf_counter()-started,process_affinity=AFFINITY,local_result_files_written=0)
 dump(OUT/'GAUGE_PROFILE77_V2.json',output);print(json.dumps({k:v for k,v in output.items() if k not in ['profiles','artifacts','process_affinity']}),flush=True)
if __name__=='__main__':main()

