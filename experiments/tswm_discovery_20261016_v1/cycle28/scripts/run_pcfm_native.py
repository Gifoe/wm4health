"""Pinned original example with recording-only wrappers; separate declared contract screen."""
import runtime_profile
import functools,hashlib,inspect,json,os,pathlib,sys,time
import numpy as np
import torch
ROOT=pathlib.Path(__file__).resolve().parents[1];SOURCE=ROOT/'upstream/PCFM';OUT=ROOT/'outputs/pcfm_native'
EXPECTED='746ff3eb69f9fcaa0e934f13b5afda17be7591a987eb64c671fa05b1d87b6755'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2),encoding='utf-8')
def to_numpy(x):return x.detach().cpu().numpy().copy() if torch.is_tensor(x) else np.asarray(x).copy()
def main():
 assert not OUT.exists();OUT.mkdir();started=time.monotonic();torch.set_num_threads(1);torch.cuda.reset_peak_memory_stats()
 manifest=json.loads((ROOT/'outputs/source/pcfm_manifest.json').read_text())
 for r in manifest['files']:assert sha(SOURCE/r['path'])==r['sha256']
 weight=ROOT/'outputs/external/PCFM/model.safetensors';assert weight.stat().st_size==211418848 and sha(weight)==EXPECTED
 sys.path.insert(0,str(SOURCE));os.environ['MPLBACKEND']='Agg'
 import model as native
 import example
 calls=[];model_holder=[]
 original_loader=native.PCFMPredictor.from_pretrained.__func__
 @classmethod
 def loader(cls,*args,**kwargs):
  result=original_loader(cls,*args,**kwargs);model_holder.append(result);return result
 native.PCFMPredictor.from_pretrained=loader
 for method in ['what_if','forecast']:
  original=getattr(native.PCFMPredictor,method)
  def wrap(function,method):
   @functools.wraps(function)
   def recorded(self,*args,**kwargs):
    index=len(calls);row=dict(index=index,method=method,caller=inspect.currentframe().f_back.f_code.co_name,kwargs={})
    calls.append(row);arrays={'input':to_numpy(args[0])}
    if len(args)>1:row['positional_tail']=list(args[1:])
    for name,value in kwargs.items():
     if torch.is_tensor(value) or isinstance(value,np.ndarray):arrays['kw_'+name]=to_numpy(value);row['kwargs'][name]='array:kw_'+name
     else:row['kwargs'][name]=value
    output=function(self,*args,**kwargs);arrays['output']=to_numpy(output)
    path=OUT/f'call{index:03d}_{method}.npz';np.savez_compressed(path,**arrays);row.update(path=path.name,sha256=sha(path),shape=list(arrays['output'].shape))
    write(OUT/'calls.json',calls);return output
   return recorded
  setattr(native.PCFMPredictor,method,wrap(original,method))
 # Preserve the native main's RNG,order,hyperparameters and all five demonstrations.
 os.chdir(OUT);sys.argv=[str(SOURCE/'example.py'),str(weight)];example.main();native_count=len(calls)
 predictor=model_holder[0];torch.cuda.synchronize()
 # The first recorded input is the unchanged window used by native demo1.
 window=np.load(OUT/calls[0]['path'])['input'];context=example.CONTEXT;h=example.HORIZON
 plans=dict(OFF=-np.ones(h),ON=np.ones(h),PULSE32=np.r_[np.ones(32),-np.ones(h-32)],
  DELAY32=np.r_[-np.ones(32),np.ones(h-32)],DELAY64=np.r_[-np.ones(64),np.ones(h-64)],DELAY96=np.r_[-np.ones(96),np.ones(h-96)])
 screen=[]
 for glue in [True,False]:
  for name,plan in plans.items():
   w=window.copy();w[0,context:]=plan
   q=predictor.what_if(w,context_length=context,masked_nodes=[1],graph=[(0,1,48.)],glue=glue,return_quantiles=True)
   index=len(calls)-1;assert np.isfinite(to_numpy(q)).all()
   screen.append(dict(plan=name,glue=glue,call_index=index))
 # Verify masked future values are not used as inputs to the prediction.
 altered=window.copy();altered[1,context:]=123456.;altered[0,context:]=-1
 replay=predictor.what_if(altered,context_length=context,masked_nodes=[1],graph=[(0,1,48.)],glue=False,return_quantiles=True)
 off=next(r for r in screen if r['plan']=='OFF' and not r['glue']);reference=np.load(OUT/calls[off['call_index']]['path'])['output']
 placeholder_error=float(np.max(np.abs(to_numpy(replay)-reference)));assert placeholder_error<2e-5,placeholder_error
 torch.cuda.synchronize()
 for r in manifest['files']:assert sha(SOURCE/r['path'])==r['sha256']
 result=dict(status='completed',source_revision=manifest['revision'],source_files_verified=len(manifest['files']),weights_sha256=EXPECTED,
  parameters=sum(p.numel() for p in predictor.model.parameters()),native_example_calls=native_count,n_recorded_calls=len(calls),screen=screen,
  placeholder_max_abs_error=placeholder_error,elapsed_s=time.monotonic()-started,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
  environment=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,device=torch.cuda.get_device_name(),threads=torch.get_num_threads()),
  source_manifest_sha256=sha(ROOT/'outputs/source/pcfm_manifest.json'),plan_sha256=sha(ROOT/'PCFM_EXECUTION_PLAN.md'),script_sha256=sha(pathlib.Path(__file__)),
  scope='Untouched native five-example execution plus separately labeled single-authored-window contract screen;not independent real-data or paper-table reproduction',
  new_training=False,accepted_final_directions=0)
 write(OUT/'results.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
