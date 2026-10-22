"""NumPy float64 metric replay and independent original-data normalization audit."""
import pathlib,json,hashlib,datetime
import torch,numpy as np
torch.set_num_threads(2);ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'native_verification_v1';OUT.mkdir(exist_ok=False)
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
result=read(ROOT/'native_recipe_v1/RESULTS_V1.json');assert read(ROOT/'recipe_execution_v1/exit.json')['exit_code']==0;acq=read(ROOT/'acquisition_recovery_v1/RESULTS_V1.json');src=pathlib.Path(acq['source_root']);q=read(ROOT/'native_recipe_v1/DATA_QUALIFICATION_V1.json');data=pathlib.Path(q['data_root']);ref=read(src/'mocap/gen_data/fingerprints.json');splits=read(src/'mocap/gen_data/splits.json');raw=read(ROOT/'native_recipe_v1/RAW_DATA_V1.json')
for x in raw['rows']:assert sha(data/'raw'/(x['trial']+'.amc'))==x['sha256']==ref['raw'][x['trial']]
checks=[];sets={};normalization=[]
for split,parts in splits.items():
 for part,names in parts.items():
  d=torch.load(data/split/f'data_{part}.pt',map_location='cpu',weights_only=True);assert d['trials']==names;sets[(split,part)]=d
  for k,v in d.items():
   if torch.is_tensor(v):assert hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest()==ref[split+'/'+part][k]['sha256'];checks.append(split+'/'+part+'/'+k)
 x=sets[(split,'train')]['data'].numpy()[:,::2].reshape(-1,50).astype(np.float64);mean=x.mean(0);std=x.std(0);std[std<1e-8]=1.
 for part in parts:
  d=sets[(split,part)];m=float(np.max(abs(mean.astype(np.float32)-d['mean'].numpy())));s=float(np.max(abs(std.astype(np.float32)-d['std'].numpy())));assert m<=1e-6 and s<=1e-6;normalization.append(dict(split=split,part=part,mean_max=m,std_max=s))
metric_rows=[];groups={};parity_failures=[]
for cell in result['cells']:
 assert sha(cell['payload_path'])==cell['payload_sha256'];m=cell['metrics'];split=cell['split'];d=sets[(split,'test')];expected=((d['data'][:,::2]-d['mean'])/d['std']).numpy()
 with np.load(cell['payload_path'],allow_pickle=False) as z:
  x=z['x'].astype(np.float64);pred=z['x_pred'].astype(np.float64);assert np.array_equal(z['x'],expected) and x.shape==(5 if split=='ID' else 7,125,50);assert np.isfinite(pred).all()
  values=dict(pred_mse=float(((x[:,75:]-pred[:,75:])**2).sum(-1).mean()))
  if cell['model']=='meta_ilad':values.update(obs_mse=float(((x[:,:75]-pred[:,:75])**2).sum(-1).mean()),recon_mse=float(((x[:,:73]-z['x_recon'].astype(np.float64))**2).sum(-1).mean()))
  else:values['recon_mse']=float(((x[:,:75]-pred[:,:75])**2).sum(-1).mean())
 for key,value in values.items():
  error=abs(value-m[key]);assert error<=1e-5+2e-6*abs(m[key]);metric_rows.append(dict(cell=cell['name'],metric=key,float64=value,native=m[key],absolute_error=error))
 groups.setdefault((cell['model'],split),[]).append(m['pred_mse']);parity_failures += [dict(cell=cell['name'],**p) for p in cell['parity'] if not p['within_tolerance']]
aggregate=[dict(model=model,split=split,seeds=len(values),mean=float(np.mean(values)),sample_sd=float(np.std(values,ddof=1))) for (model,split),values in groups.items()]
for row in acq['files']:assert sha(src/row['path'])==row['sha256']
out=dict(status='NATIVE_MOCAP_DATA_AND_METRICS_INDEPENDENTLY_VERIFIED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),raw_file_checks=len(raw['rows']),tensor_fingerprint_checks=len(checks),train_only_normalization_checks=normalization,metric_checks=len(metric_rows),max_metric_error=max(x['absolute_error'] for x in metric_rows),metrics=metric_rows,aggregate=aggregate,reference_parity_failures=parity_failures,reference_parity_qualified=not parity_failures,unique_benchmark_cells=9,actual_evaluation_calls=11,new_fits=0,source_files_unchanged=True,causal_mechanism_fidelity_evaluated=False,accepted_final_directions=0)
with (OUT/'VERIFICATION_V1.json').open('x',encoding='utf-8') as f:json.dump(out,f,indent=2,allow_nan=False)
print(json.dumps({k:v for k,v in out.items() if k not in ['metrics','train_only_normalization_checks']}),flush=True)
