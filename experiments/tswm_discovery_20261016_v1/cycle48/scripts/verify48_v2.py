"""Independent raw CSV/ZIP windows and deterministic score reconstruction."""
import pathlib,sys,json,hashlib,csv,zipfile,io,datetime,calendar
R=pathlib.Path(__file__).resolve().parents[1];ROOT=R.parent;sys.path.insert(0,str(ROOT/'cycle29/scripts'));import runtime_profile
import numpy as np
from independent_model48 import replay
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
 out=R/'outputs/eval_v1';assert not (out/'verification.json').exists();data_manifest=read(R/'data/MANIFEST.json');original=read(ROOT/'cycle40/outputs/intake.json');source=ROOT/'cycle40/assets'
 for ent in original['sources']:assert sha(source/ent['file'])==ent['sha256']
 def stamp(s,fmt=None):return calendar.timegm((datetime.datetime.strptime(s,fmt) if fmt else datetime.datetime.fromisoformat(s)).timetuple())*10**9
 with (source/'ETTh1.csv').open(newline='',encoding='utf-8-sig') as f:erows=list(csv.DictReader(f))
 et=np.array([stamp(x['date']) for x in erows],dtype='int64');ey=np.array([float(x['OT']) for x in erows]);observed={}
 for half in ['a','b']:
  with zipfile.ZipFile(source/f'mpi_roof_2016{half}.zip') as archive:
   filename=next(n for n in archive.namelist() if n.lower().endswith('.csv'))
   with archive.open(filename) as f:
    for row in csv.DictReader(io.TextIOWrapper(f,encoding='latin1')):
     date=datetime.datetime.strptime(row['Date Time'],'%d.%m.%Y %H:%M:%S')
     if date.minute:continue
     key=calendar.timegm(date.timetuple())*10**9;assert key not in observed;value=float(row['T (degC)']);observed[key]=np.nan if value==-9999 else value
 jt=np.arange(min(observed),max(observed)+3600*10**9,3600*10**9,dtype='int64');jy=np.array([observed.get(int(t),np.nan) for t in jt]);sources={'ett':(et,ey),'jena':(jt,jy)};reconstructed={};max_window=0.
 for case,(times,raw) in sources.items():
  n=len(raw);bounds=[0,int(.7*n),int(.85*n),n];a=raw[:bounds[1]];mean=np.nanmean(a);sd=np.nanstd(a);normalized=(raw-mean)/sd;assert abs(mean-data_manifest['description'][case]['train_mean'])<2e-12 and abs(sd-data_manifest['description'][case]['train_std'])<2e-12
  for i,(split,stride) in enumerate([('train',3),('val',48),('eval',48)]):
   z=np.load(R/'data'/case/(split+'.npz'));valid=np.array([o for o in range(bounds[i]+96,bounds[i+1]-48+1,stride) if np.isfinite(normalized[o-96:o+48]).all()],dtype='int64');assert np.array_equal(valid,z['origins']);history=np.stack([normalized[o-96:o] for o in valid]).astype('float32')[:,:,None];future=np.stack([normalized[o:o+48] for o in valid]).astype('float32')[:,:,None];phase=(((times[valid-1]//(3600*10**9))%24)/24).astype('float32');assert np.array_equal(z['history_timestamps'],np.stack([times[o-96:o] for o in valid])) and np.array_equal(z['future_timestamps'],np.stack([times[o:o+48] for o in valid]));error=max(float(np.max(abs(history-z['history']))),float(np.max(abs(future-z['future']))));max_window=max(max_window,error);assert error<1e-6 and np.array_equal(phase,z['phase']);reconstructed[case,split]={'history':history,'future':future,'phase':phase,'origins':valid}
 for name,ent in data_manifest['files'].items():assert sha(R/'data'/name)==ent['sha256']
 summary=read(out/'summary.json');manifest=read(out/'raw_manifest.json');count=0;maximum=0.;means={};max_model=0.;bundles={}
 for ent in read(out/'model_bundles.json'):
  assert sha(out/ent['path'])==ent['sha256'];assert sha(R/'outputs/fits'/ent['model']/'selected.pt')==ent['selected_checkpoint_sha256'];bundles[ent['model']]={k:v for k,v in np.load(out/ent['path']).items()}
 for ent in manifest:
  p=out/ent['path'];assert sha(p)==ent['sha256'];z=np.load(p);original_data=reconstructed[ent['case'],'eval'];reveal=ent['reveal'];expected_history=original_data['history'] if not reveal else np.concatenate((original_data['history'],original_data['future'][:,:reveal]),axis=1)[:,-96:];expected_truth=original_data['future'][:,reveal:];expected_phase=original_data['phase'] if not reveal else (original_data['phase']+reveal/24)%1;assert np.array_equal(z['origins'],original_data['origins']) and np.allclose(z['history'],expected_history,atol=1e-6,rtol=0) and np.allclose(z['truth'],expected_truth,atol=1e-6,rtol=0) and np.allclose(z['phase'],expected_phase,atol=2e-7,rtol=0);prediction=z['prediction'].astype('float64')[:,:,0];truth=z['truth'].astype('float64')[:,:,0]
  if ent['model'] in bundles:independent=replay(ent['method'],bundles[ent['model']],z['history'],z['phase'],ent['horizon'])[:,:,0]
  elif ent['method']=='ridge':
   coef=np.load(R/'outputs/control_v1'/(ent['case']+'_ridge.npz'))['weights'];xx=np.concatenate((z['history'][:,:,0],np.sin(2*np.pi*z['phase'])[:,None],np.cos(2*np.pi*z['phase'])[:,None],np.ones((len(prediction),1))),1).astype('float64');independent=(xx@coef)[:,:ent['horizon']]
  elif ent['method']=='persistence':independent=np.repeat(z['history'][:,-1,0,None],ent['horizon'],axis=1)
  else:independent=np.tile(z['history'][:,-24:,0],(1,ent['horizon']//24))
  model_error=float(np.max(abs(prediction-independent)));max_model=max(max_model,model_error);assert np.allclose(prediction,independent,atol=3e-4,rtol=3e-4),(ent['path'],'independent model replay',model_error)
  error=prediction-truth;assert np.isfinite(error).all();values={'mse':np.mean(error**2,1),'mae':np.mean(abs(error),1),'first12_mse':np.mean(error[:,:12]**2,1),'last12_mse':np.mean(error[:,-12:]**2,1),'tail24_mse':np.mean(error[:,-24:]**2,1),'aggregate24_mse':np.mean(error.reshape(len(error),-1,24).mean(-1)**2,1),'aggregate_all_mse':np.mean(error,1)**2}
  for key,value in values.items():difference=float(np.max(abs(value-z[key])));maximum=max(maximum,difference);assert difference<2e-10,(ent['path'],key,difference)
  means[ent['model'],reveal]={key:float(value.mean()) for key,value in values.items()};count+=len(error)
 for row in summary['groups']:
  for key,value in means[row['model'],row['reveal']].items():assert abs(row[key]-value)<2e-10
 schedules={};initials={}
 for folder in (R/'outputs/fits').iterdir():
  fit=read(folder/'fit_summary.json');cfg=read(folder/'config.json');assert sha(folder/'selected.pt')==fit['selected_sha256'] and sha(folder/'last.pt')==fit['last_sha256'] and sha(folder/'schedule.npz')==fit['schedule_sha256'];assert sha(R/'scripts/common48_v3.py')==cfg['common_sha256'] and sha(R/'PROTOCOL.md')==cfg['protocol_sha256'] and sha(R/'data/MANIFEST.json')==cfg['data_manifest_sha256'];schedules.setdefault((fit['case'],fit['seed']),set()).add(fit['schedule_sha256']);initials[fit['case'],fit['seed'],fit['method']]=fit['initial_weight_checksum']
 assert len(schedules)==4 and all(len(s)==1 for s in schedules.values())
 for case,seed in schedules:assert initials[case,seed,'fixed_dde']==initials[case,seed,'learned_dde']
 unique=sum(len(reconstructed[case,'eval']['origins']) for case in ['ett','jena']);assert len(manifest)==44 and count==unique*22 and count==summary['model_origin_reveal_entries'];result={'status':'PASS_REAL_PROCESS_MEMORY_SCREEN','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'raw_groups':44,'model_origin_reveal_entries':count,'unique_forecast_origins':unique,'real_processes':2,'origins_not_assumed_iid':True,'max_independent_window_error':max_window,'max_score_arithmetic_error':maximum,'max_independent_model_forecast_error':max_model,'independent_model_replay':'NumPy GRU gates, ReLU field, Heun method-of-steps, calendar and history interpolation','source_csv_zip_parser':'stdlib csv/zipfile/datetime, not pandas or cached windows','per_case_seed_schedules_identical':True,'fixed_learned_initial_weights_identical':True,'summary_sha256':sha(out/'summary.json'),'raw_manifest_sha256':sha(out/'raw_manifest.json'),'source_sha256':sha(pathlib.Path(__file__))};(out/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
