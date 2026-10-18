"""Exact native pretrained model; diagnostics only, all arrays remain server G."""
import os
os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='4'
import pathlib,json,hashlib,datetime,sys,types,importlib,inspect,ast,textwrap,difflib,time
import numpy as np,torch,sklearn,safetensors
from safetensors.torch import load_file
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs'
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  while b:=f.read(8*1024*1024):h.update(b)
 return h.hexdigest()
def dump(p,obj):
 with p.open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,allow_nan=False)
assert sys.flags.isolated and not (OUT/'INTERFACE66_V2.json').exists()
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest');torch.manual_seed(2021);np.random.seed(2021)
manifest=read(OUT/'MODEL_FILES66_V1.json');base=pathlib.Path(manifest['server_directory'])
assert manifest['revision']=='3d1281c9994140056edfbbdedab57f2e6493dfaa'
for row in manifest['files']:assert sha(row['server_path'])==row['sha256']
assert sha(base/'model.safetensors')=='44ddf4118b4b027a5b726ce7368439e0fa28fd3c331909149befac2087daad11'
pkg=types.ModuleType('normwear66_native');pkg.__path__=[str(base)];sys.modules[pkg.__name__]=pkg
mod=importlib.import_module(pkg.__name__+'.normwear2');cfg=read(base/'config.json');args={k:v for k,v in cfg.items() if k in inspect.signature(mod.NormWear2).parameters}
model=mod.NormWear2(**args);weights=load_file(str(base/'model.safetensors'),device='cpu');assert set(weights)==set(model.state_dict()), 'Pinned checkpoint has bare native keys; no prefix conversion' 
state=weights
# Header audit establishes exact bare native keys; strict load follows.
result=model.load_state_dict(state,strict=True);assert not result.missing_keys and not result.unexpected_keys
assert all(torch.equal(model.state_dict()[k],v) for k,v in state.items())
assert all(torch.isfinite(v).all().item() for v in state.values())
state_elements=sum(v.numel() for v in state.values());parameters=sum(p.numel() for p in model.parameters());buffers=sum(p.numel() for p in model.buffers())
device='cuda';assert torch.cuda.is_available();model=model.to(device).eval();del weights,state
print(json.dumps(dict(status='STRICT_NATIVE_WEIGHTS_LOADED',state_elements=state_elements,parameters=parameters,buffers=buffers,device=torch.cuda.get_device_name())),flush=True)
t=torch.arange(160,device=device,dtype=torch.float32)
x=torch.stack([.7*torch.sin(t*.13)+.2*torch.cos(t*.037),.6*torch.cos(t*.071)+.3*torch.sin(t*.19)],dim=-1)[None]
arrays={'fixture':x.cpu().numpy()};records=[];orig_sim=model.simulate;orig_encoder=model.forward_encoder;orig_source=inspect.getsource(mod.NormWear2.simulate)
tree=ast.parse(textwrap.dedent(orig_source));count=0
for node in ast.walk(tree):
 if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Subscript) and isinstance(node.targets[0].value,ast.Name) and node.targets[0].value.id=='pad_context_tensor':
  # Only the stored recurrent decode update, not initialization.
  if isinstance(node.value,ast.Subscript) and isinstance(node.value.value,ast.Name) and node.value.value.id=='dec_out':
   for sub in [node.targets[0],node.value]:
    assert isinstance(sub.slice,ast.Tuple) and len(sub.slice.elts)==3 and isinstance(sub.slice.elts[1],ast.Slice)
    sub.slice.elts[1]=ast.Name(id='non_visible_channel',ctx=ast.Load());count+=1
assert count==2;ast.fix_missing_locations(tree);repair_source=ast.unparse(tree)
namespace=dict(mod.__dict__);exec(compile(tree,'<authored_known_covariate_clamp66_v1>','exec'),namespace);repaired_sim=types.MethodType(namespace['simulate'],model)
diff=''.join(difflib.unified_diff(textwrap.dedent(orig_source).splitlines(True),(repair_source+'\n').splitlines(True),fromfile='native simulate',tofile='AST formatting plus two indexed-channel changes'))
with (OUT/'KNOWN_COVARIATE_CLAMP_DIFF66_V2.txt').open('x',encoding='utf-8') as f:f.write(diff)
def run_sim(name,inp,method):
 before=inp.clone();trace=[]
 def encoder_trace(this,z,*a,**kw):trace.append(z.detach().cpu().numpy().copy());return orig_encoder(z,*a,**kw)
 model.forward_encoder=types.MethodType(encoder_trace,model);torch.manual_seed(2021);np.random.seed(2021);torch.cuda.synchronize();start=time.perf_counter()
 try:
  with torch.inference_mode():y=method(inp,all_visible_length=128,non_visible_channel=[0],ar_step=16)
 finally:model.forward_encoder=orig_encoder
 torch.cuda.synchronize();assert torch.isfinite(y).all();arrays[name]=y.cpu().numpy()
 normalized_u=(before[0,:,1]-before[0,:,1].mean())/(before[0,:,1].std()+1e-8)
 assert len(trace)==2
 records.append(dict(name=name,shape=list(y.shape),finite=True,seconds=time.perf_counter()-start,input_mutated=bool(not torch.equal(before,inp)),known_future_output_max_abs_change=float((y[:,:160,1]-before[:,:,1]).abs().max()),second_encoder_known_first_future_block_max_abs_change=float(np.max(np.abs(trace[1][0,1,128:144]-normalized_u[128:144].cpu().numpy())))))
 for i,z in enumerate(trace):arrays[name+'_encoder_trace'+str(i)]=z
 return y
with torch.inference_mode():pred=model.predict(x[:,:128].clone(),prediction_length=32,max_pred_length=16)
assert pred.shape==(1,32,2) and torch.isfinite(pred).all();arrays['native_predict']=pred.cpu().numpy()
native=run_sim('native_simulate',x.clone(),orig_sim)
hidden=x.clone();hidden[:,128:,0]+=111;hidden_result=run_sim('native_hidden_perturbed',hidden,orig_sim)
hidden_error=float((native-hidden_result).abs().max());assert hidden_error==0
clamped=run_sim('clamped_simulate',x.clone(),repaired_sim)
future=x.clone();future[:,144:,1]+=2*torch.sin(torch.arange(16,device=device,dtype=torch.float32)*.3+.2)
changed=run_sim('native_late_covariate_perturbed',future,orig_sim)
anticipation=float((native[:,128:144,0]-changed[:,128:144,0]).abs().max())
cfg_storage=read(ROOT.parent/'STORAGE_LOCATION_V1.json');art=pathlib.Path(cfg_storage['artifact_root'])/'cycle66';art.mkdir(exist_ok=True,parents=True);ap=art/'INTERFACE_ARRAYS66_V2.npz';assert not ap.exists();np.savez_compressed(ap,**arrays)
for row in manifest['files']:assert sha(row['server_path'])==row['sha256']
obj=dict(utc=utc(),status='NATIVE_PRETRAINED_INTERFACE_QUALIFIED_DIAGNOSTICS_ONLY',source_revision=manifest['revision'],model_files_receipt_sha256=sha(OUT/'MODEL_FILES66_V1.json'),weight_sha256=sha(base/'model.safetensors'),checkpoint_key_layout='Bare native keys verified by WEIGHT_HEADER66_V1; no key conversion',loader_correction='V1 assertion wrongly assumed wrapper-prefixed weights and stopped before inference; V2 uses all exact native keys',strict_loaded_tensor_values_equal=True,strict_missing_keys=[],strict_unexpected_keys=[],parameters=parameters,state_elements=state_elements,buffers=buffers,config=args,wrapper='Thin HF wrapper bypassed; exact full native class/config/state used via relative-import package namespace',environment=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,sklearn=sklearn.__version__,safetensors=safetensors.__version__,numpy_origin=np.__file__,torch_origin=torch.__file__,device=torch.cuda.get_device_name(),isolated=True),predict_shape=list(pred.shape),hidden_future_perturbation_max_abs_change=hidden_error,late_covariate_perturbation_earlier_response_max_abs_change=anticipation,simulate_runs=records,clamp_AST_subscript_changes=count,repair_diff_sha256=sha(OUT/'KNOWN_COVARIATE_CLAMP_DIFF66_V2.txt'),array_artifact=str(ap),array_sha256=sha(ap),protocol_sha256=sha(ROOT/'INTERFACE_PROTOCOL66_V1.md'),script_sha256=sha(pathlib.Path(__file__)),performance_benchmark=False,causal_identification_test=False,accepted_final_directions=0)
dump(OUT/'INTERFACE66_V2.json',obj);print(json.dumps(obj),flush=True)
