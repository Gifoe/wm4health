"""Execute pinned author's minimal example and qualify API semantics, not accuracy."""
import sys,pathlib,os,json,hashlib,shutil,datetime,time,subprocess,random
R=pathlib.Path(__file__).resolve().parents[1];B=R.parent
sys.path.insert(0,str(B/'cycle29/scripts'));import runtime_profile
os.environ['MPLBACKEND']='Agg';sys.stdout.reconfigure(encoding='utf-8')
O=R/'outputs/native_v1';assert not O.exists();O.mkdir(parents=True)
SOURCE=B/'cycle40/source_archive/lifts';S=R/'upstream/lifts';S.mkdir(parents=True,exist_ok=True);manifest=[]
for entry in json.loads((SOURCE/'CACHE_RECEIPTS.json').read_text())['files']:
 src=SOURCE/entry['path'];raw=src.read_bytes();assert hashlib.sha256(raw).hexdigest()==entry['sha256']
 dst=S/entry['path'];dst.parent.mkdir(parents=True,exist_ok=True)
 if dst.exists():assert dst.read_bytes()==raw
 else:dst.write_bytes(raw)
 manifest.append({'path':entry['path'],'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)})
for entry in json.loads((R/'source_supplement/RECEIPTS.json').read_text())['files']:
 raw=(R/'source_supplement'/entry['file']).read_bytes();assert hashlib.sha256(raw).hexdigest()==entry['sha256']
 dst=S/entry['file'];dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(raw)
 manifest.append({'path':entry['file'],'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)})
(O/'source_manifest.json').write_text(json.dumps({'commit':'e42430980e51bebf2e1b54d1beddeda7a31dee51','files':manifest},indent=2))
sys.path.insert(0,str(S/'src'))
import numpy as np
import torch
from lifts import LIFTSImputer,make_case1_dataset
from loader.case1 import generate_case1
torch.set_num_threads(1)
def seed(v):random.seed(v);np.random.seed(v);torch.manual_seed(v);torch.cuda.manual_seed_all(v)
def statehash(model):
 h=hashlib.sha256()
 for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()
seed(42)
native=generate_case1(num_sample_train=80,num_sample_test=20,p=20,T=48,d=5,obs_rate=.5,seed=42,mm='mcar')
data=make_case1_dataset(num_sample_train=80,num_sample_test=20,p=20,T=48,d=5,obs_rate=.5,seed=42,mm='mcar')
assert np.array_equal(data['X_train_full'],torch.stack(native[3][0]).numpy())
assert np.array_equal(data['mask_train'],torch.stack(native[1][0]).numpy())
np.savez_compressed(O/'fixture.npz',**data,Z_train=torch.stack(native[2][0]).numpy(),Z_check=torch.stack(native[2][1]).numpy(),phi=native[4].numpy(),F=native[5].numpy(),Sigma_p=native[6][3].numpy(),Sigma_d=native[6][4].numpy(),W_mask=native[6][1].numpy(),b_mask=native[6][2].numpy())
imputer=LIFTSImputer(p=20,T=48,d=5,obs_rate=.5,total_loops=5,batch_size=32,n_samples=10,lambda_selfmask=5.,lambda_enc=1.,enc_warmup=2,seed=42)
gpu_before=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
torch.cuda.reset_peak_memory_stats();a=time.perf_counter();imputer.fit(data['X_train'],data['mask_train'],verbose=True);torch.cuda.synchronize();elapsed=time.perf_counter()-a
torch.save({'model_state_dict':imputer.model.state_dict(),'config':imputer.config,'mean_scaler':imputer.mean_scaler,'scaler':imputer.scaler,'source_commit':'e42430980e51bebf2e1b54d1beddeda7a31dee51'},O/'native_minimal.pt')
(O/'native_config.json').write_text(json.dumps(imputer.config,indent=2))
before=statehash(imputer.model);seed(4242)
samples=imputer.impute_samples(data['X_test'],data['mask_test'],n_samples=10)
seed(4343);forecast=imputer.forecast_samples(data['X_test'],data['mask_test'],n_samples=10)
assert samples.shape==forecast.shape==(20,10,20,48) and np.isfinite(samples).all() and np.isfinite(forecast).all()
observed=data['mask_test'].astype(bool);hard=float(np.max(np.abs(samples-data['X_test_full'][:,None])[np.broadcast_to(observed[:,None],samples.shape)]))
assert hard<2e-6,hard
checks=[];x=data['X_test'][:2].copy();mask=data['mask_test'][:2].copy();corrupt=x.copy();corrupt[mask==0]=123.456
for name,fn in [('imputation',imputer.impute_samples),('forecast',imputer.forecast_samples)]:
 seed(4444);u=fn(x,mask,n_samples=4)
 seed(4444);v=fn(x,mask,n_samples=4)
 replay=float(np.abs(u-v).max());assert replay<2e-6,replay
 seed(4444);w=fn(corrupt,mask,n_samples=4)
 missing=float(np.abs(u-w).max());assert missing<2e-6,(name,missing)
 checks.append({'api':name,'replay_max_abs':replay,'unobserved_value_corruption_max_abs':missing})
# Causal forecast is characterized separately from the smoothing/imputation API.
changed=x.copy();changed[:,:,24:]=np.where(mask[:,:,24:]>0,123.456,changed[:,:,24:])
seed(4545);u=imputer.forecast_samples(x,mask,n_samples=4);seed(4545);v=imputer.forecast_samples(changed,mask,n_samples=4)
causal=float(np.max(np.abs(u[:,:,:,:24]-v[:,:,:,:24])));assert causal<2e-6,causal
assert statehash(imputer.model)==before
np.savez_compressed(O/'native_samples.npz',imputation=samples,forecast=forecast,check_X=data['X_test_full'],check_mask=data['mask_test'])
report={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'PASS','source_files':len(manifest),'model_parameters':sum(p.numel() for p in imputer.model.parameters()),'example':'Author case1_minimal settings, not paper headline reproduction','paper_main_lambda_enc':0,'executed_example_lambda_enc':1,'training_loops':5,'nominal_minibatch_updates':15,'n_train':80,'n_interface_check':20,'gpu_context_before':gpu_before,'runtime_seconds':elapsed,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,'checks':checks,'hard_replacement_max_error':hard,'forecast_earlier_half_future_corruption_max_error':causal,'mean_imputation_draw_std':float(samples.std(axis=1).mean()),'mean_forecast_draw_std':float(forecast.std(axis=1).mean()),'evaluation_weights_unchanged':True,'fixture_sha256':hashlib.sha256((O/'fixture.npz').read_bytes()).hexdigest(),'checkpoint_sha256':hashlib.sha256((O/'native_minimal.pt').read_bytes()).hexdigest(),'samples_sha256':hashlib.sha256((O/'native_samples.npz').read_bytes()).hexdigest(),'test_scored':False,'no_accuracy_or_convergence_claim':True}
(O/'qualification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
