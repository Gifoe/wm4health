"""Isolate decoder arithmetic from input dependence without changing any benchmark."""
import pathlib,sys,json,hashlib,datetime,importlib.util,traceback
import torch,numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'pendulum_numeric_qualification_v1';OUT.mkdir(exist_ok=False)
SRC=pathlib.Path(r'G:\ICLR Omni-eeg cache\tswm_discovery_20261016_v1\caches\meta_ilad_native_20261023_v1\source/pendulum')
ART=pathlib.Path(r'G:\ICLR Omni-eeg cache\tswm_discovery_20261016_v1\artifacts\meta_ilad_native_20261023_v1\pendulum_numeric_qualification_v1');ART.mkdir(exist_ok=False)
def dump(n,x):
 with (OUT/n).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def digest(model):
 h=hashlib.sha256()
 for k,v in sorted(model.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
try:
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 sys.path.insert(0,str(SRC/'meta_ilad'));from model import iLaD
 from dataloader import MyDataLoader
 torch.manual_seed(1000);model=iLaD('cuda').cuda();model.load_state_dict(torch.load(SRC/'checkpoints/meta_ilad_s1000.pt',weights_only=True,map_location='cuda'));model.eval();before=digest(model)
 data=json.loads((ROOT/'pendulum_data_generation_v1/RESULTS_V1.json').read_text());path=next(x['path'] for x in data['sets'] if x['split']=='test')
 batch,_=next(iter(MyDataLoader(path,12,128,shuffle=False,seed=42,device='cuda')));x,ctx=batch[:2];x=x[:,:40];ctx=ctx[:,:-1,:40];shadow=x.clone();shadow[:,3:]+=17
 original_flag=torch.backends.cudnn.deterministic;cpu_rng=torch.get_rng_state();gpu_rng=torch.cuda.get_rng_state()
 try:
  torch.backends.cudnn.deterministic=True
  with torch.no_grad():a=model.evaluate_TG(x,ctx)
  torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(gpu_rng)
  with torch.no_grad():b=model.evaluate_TG(shadow,ctx)
 finally:torch.backends.cudnn.deterministic=original_flag
 errors={str(i):float((v-w).abs().max()) for i,(v,w) in enumerate(zip(a,b))};assert all(v==0 for v in errors.values()),errors
 with torch.no_grad():d1=model.decode(a[1]);d2=model.decode(a[1])
 decoder_repeat_error=float((d1-d2).abs().max());assert digest(model)==before
 path=ART/'shadow_arrays.npz'
 with path.open('xb') as f:np.savez_compressed(f,original_query=x.cpu().numpy(),modified_query=shadow.cpu().numpy(),original_prediction=a[0].cpu().numpy(),modified_prediction=b[0].cpu().numpy(),original_latent=a[1].cpu().numpy(),modified_latent=b[1].cpu().numpy(),original_environment=a[2].cpu().numpy(),modified_environment=b[2].cpu().numpy(),decoder_repeat1=d1.cpu().numpy(),decoder_repeat2=d2.cpu().numpy())
 result=dict(status='EXACT_BOUNDARY_INVARIANCE_UNDER_DETERMINISTIC_DECODER_QUALIFIED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),deterministic_scope=True,original_cudnn_deterministic=original_flag,restored_flag=torch.backends.cudnn.deterministic,errors=errors,decoder_repeat_max_error_under_original_flag=decoder_repeat_error,original_default_pixel_gate_failed=True,original_default_latent_and_environment_errors=0.,new_native_boundary_calls=2,new_decoder_only_calls=2,benchmark_calls=0,new_fits=0,model_state_before=before,model_state_after=digest(model),payload_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),local_payload_sync=False)
 dump('RESULTS_V1.json',result);print(json.dumps(result),flush=True)
except Exception as exc:
 dump('FAILURE_V1.json',dict(exception_type=type(exc).__name__,exception=str(exc),traceback=traceback.format_exc()));raise
