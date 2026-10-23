"""Execute four original data recipes with the qualified native forward kernels."""
import pathlib,sys,json,hashlib,datetime,time,os,shutil,types,gc,traceback
import torch,numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).parent));from native_renderer_loader_v1 import load
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'pendulum_data_generation_v1';OUT.mkdir(exist_ok=False)
DATA=pathlib.Path(r'G:\ICLR Omni-eeg cache\tswm_discovery_20261016_v1\data\meta_ilad_native_20261023_v1\pendulum_native_nvrtc_v1');DATA.mkdir(exist_ok=False)
def sha_file(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(n,x):
 with (OUT/n).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def event(stage,**kw):
 row=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),stage=stage,**kw)
 with (OUT/'trace.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
 print(json.dumps(row),flush=True)
try:
 assert json.loads((ROOT/'renderer_geometry_qualification_v2/RESULTS_V1.json').read_text())['full_vs_chunked_max_error']==0
 intake=json.loads((ROOT/'pendulum_acquisition_recovery_v1/RESULTS_V1.json').read_text())
 src=pathlib.Path(intake['source_root'])
 for receipt in [intake,json.loads((ROOT/'physics_qualification_v1/ACQUISITION_V1.json').read_text())]:
  for row in receipt['files']:assert sha_file(src/row['path'])==row['sha256']
 assert shutil.disk_usage(DATA).free>16*1024**3
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 native=load(ROOT);gen=src/'pendulum/gen_data';os.chdir(gen);sys.path.insert(0,str(gen))
 import generate_data as original
 from fingerprint import fingerprint,check
 refs=json.loads((gen/'fingerprints.json').read_text());manifest=[];started=time.monotonic()
 dump('PLAN_V1.json',dict(source_revision=intake['source_revision'],data_root=str(DATA),protocol_sha256=sha_file(ROOT/'PENDULUM_DATA_GENERATION_PROTOCOL_V1.md'),batch_sequences=250,frames_per_sequence=60,original_axis_calls_unchanged=True,model_evaluations=0,new_fits=0))
 def qualify(split,path,data):
  fps={};n=len(data['xt']);assert data['xt'].shape==(n,60,2) and data['yt'].shape==(n,60,32,32) and data['params'].shape==(n,2) and data['label'].shape==(n,1)
  for key,t in data.items():
   assert t.dtype==torch.float32
   for chunk in t.split(250):
    assert torch.isfinite(chunk).all(),(split,key)
    if key=='yt':assert chunk.min()>=0 and chunk.max()<=1
   fp=fingerprint(t);ref=refs[split][key];rel=abs(fp['sum']-ref['sum'])/max(abs(ref['sum']),1e-12)
   if key in ['params','label']:assert fp['sha256']==ref['sha256'],(split,key)
   if key=='yt':assert rel<=1e-4,(split,key,rel)
   fps[key]=dict(generated=fp,reference=ref,exact_match=fp['sha256']==ref['sha256'],relative_sum_error=rel)
  # A separate deserialize/fingerprint pass guards storage and format adaptation.
  reopened=torch.load(path,weights_only=True,mmap=True,map_location='cpu');assert set(reopened)==set(data)
  for key,t in reopened.items():assert fingerprint(t)['sha256']==fps[key]['generated']['sha256']
  row=dict(split=split,sequences=n,path=str(path),bytes=path.stat().st_size,file_sha256=sha_file(path),tensors=fps,storage_roundtrip_exact=True,functional_sanity_passed=True,source_bitwise_all_tensors=all(x['exact_match'] for x in fps.values()))
  manifest.append(row);dump('SET_'+split.upper()+'_V1.json',row);event('set_complete',split=split,sequences=n,bytes=path.stat().st_size,image_sum_relative_error=fps['yt']['relative_sum_error'],source_bitwise_all_tensors=row['source_bitwise_all_tensors'])
 for split,cfg in original.SPLITS.items():
  event('set_started',split=split);data=original.generate(cfg,'cuda:0',batch_size=250,render=True);check(data,split);path=DATA/cfg['file']
  with path.open('xb') as f:torch.save(data,f)
  qualify(split,path,data);del data;gc.collect();torch.cuda.empty_cache()
 import make_trainenv_test as control
 path=DATA/'data_test_trainenv.pt';assert not path.exists();event('set_started',split='trainenv')
 control.main(types.SimpleNamespace(train=str(DATA/'data_train.pt'),out=str(path),n_init=20,seed=5000,device='cuda:0',batch=250,verify=0,no_render=False))
 data=torch.load(path,weights_only=True,mmap=True,map_location='cpu');qualify('trainenv',path,data);del data;gc.collect()
 for receipt in [intake,json.loads((ROOT/'physics_qualification_v1/ACQUISITION_V1.json').read_text())]:
  for row in receipt['files']:assert sha_file(src/row['path'])==row['sha256']
 result=dict(status='ALL_FOUR_ORIGINAL_RECIPE_INPUT_SETS_FUNCTIONALLY_QUALIFIED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),sets=manifest,total_sequences=sum(x['sequences'] for x in manifest),elapsed_seconds=time.monotonic()-started,source_revision=intake['source_revision'],original_legacy_stack_installed=False,binding='unchanged CUDA kernels via NVRTC; original renderer Python; restricted original OBJ parser',model_evaluations=0,new_fits=0,local_payload_sync=False)
 dump('RESULTS_V1.json',result);native.backend.module.close();event('data_generation_complete',total_sequences=result['total_sequences'],elapsed_seconds=result['elapsed_seconds'])
except Exception as e:
 dump('FAILURE_V1.json',dict(exception_type=type(e).__name__,exception=str(e),traceback=traceback.format_exc()));raise
