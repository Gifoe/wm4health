"""One changed-critical-path qualification: launch context, native kernels and alpha geometry."""
import pathlib,sys,hashlib,json,datetime,traceback,types,importlib.util,ctypes as C
import torch,numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).parent))
from nvrtc_native_forward_v1 import NVRTCModule,NativeForward,P,I
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'renderer_kernel_qualification_v1';OUT.mkdir(exist_ok=False)
SRC=pathlib.Path(r'G:\ICLR Omni-eeg cache\tswm_discovery_20261016_v1\caches\gradsim_abe8008_native_renderer_v1')
def sha(b):return hashlib.sha256(b).hexdigest()
def dump(n,x):
 with (OUT/n).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
try:
 acq=json.loads((ROOT/'renderer_acquisition_v1/RESULTS_V1.json').read_text())
 for row in acq['files']:assert sha((SRC/row['path']).read_bytes())==row['sha256']
 torch.set_num_threads(1);torch.cuda.init()
 smoke='extern "C" __global__ void affine(const float *x,float *y,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=2*x[i]+1;}'
 m=NVRTCModule(smoke,['&affine']);stream=torch.cuda.Stream()
 with torch.cuda.stream(stream):
  x=torch.arange(1024,device='cuda',dtype=torch.float32);y=torch.empty_like(x);m.launch('&affine',4,256,[P(x.data_ptr()),P(y.data_ptr()),I(len(x))]);expected=2*x+1
 stream.synchronize();assert torch.equal(y,expected);m.close()
 backend=NativeForward(SRC)
 pkg=types.ModuleType('_pinned_native_softras');pkg.__path__=[str(SRC/'gradsim/renderutils')];sys.modules[pkg.__name__]=pkg
 sub=types.ModuleType(pkg.__name__+'.cuda');sub.soft_rasterize_cuda=backend;sys.modules[sub.__name__]=sub
 spec=importlib.util.spec_from_file_location(pkg.__name__+'.soft_rasterize',SRC/'gradsim/renderutils/soft_rasterize.py');native=importlib.util.module_from_spec(spec);sys.modules[spec.name]=native;spec.loader.exec_module(native)
 fixtures=torch.tensor([[[[-.73,-.61,3.],[.81,-.47,3.],[-.11,.86,3.]]],[[[-.52,-.78,5.],[.67,-.41,5.],[.24,.74,5.]]]],dtype=torch.float32,device='cuda')
 texture=torch.ones(2,1,1,3,device='cuda');h=17
 image=native.soft_rasterize(fixtures,texture,h,[0,0,0],dist_func='hard',aggr_func_rgb='hard',aggr_func_alpha='hard')
 torch.cuda.synchronize();actual=image[:,-1].cpu().double().numpy();expected=[]
 for triangle in fixtures.cpu().double().numpy()[:,0]:
  matrix=np.concatenate([triangle[:,:2].T,np.ones((1,3))],axis=0);inv=np.linalg.inv(matrix);mask=np.empty((h,h))
  for row in range(h):
   for col in range(h):
    q=np.array([(2*col+1-h)/h,(2*(h-1-row)+1-h)/h,1.]);w=inv@q;mask[row,col]=float(np.all(w>=0) and np.all(w<=1))
  expected.append(mask)
 err=float(np.max(np.abs(actual-np.stack(expected))));assert err==0
 # Exercise the original euclidean/product branch; full soft geometry validation follows separately.
 soft=native.soft_rasterize(fixtures,texture,h,[0,0,0]);torch.cuda.synchronize();assert torch.isfinite(soft).all() and torch.all((soft[:,-1]>=0)&(soft[:,-1]<=1))
 rows=dict(status='NVRTC_CONTEXT_STREAM_AND_NATIVE_FORWARD_KERNELS_QUALIFIED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_revision=acq['source_revision'],source_files_checked=len(acq['files']),original_kernel_sha256=backend.original_source_sha256,compilation_unit_sha256=backend.module.source_sha256,ptx_sha256=backend.module.ptx_sha256,compile_options=backend.module.options,compile_log=backend.module.compile_log,smoke_values=1024,nondefault_stream_smoke_exact=True,hard_triangle_pixels=actual.size,hard_triangle_max_error=err,default_soft_forward_finite=True,renderer_complete=False,model_inputs_qualified=False,model_evaluations=0,new_fits=0,local_payload_sync=False)
 dump('RESULTS_V1.json',rows);backend.module.close();print(json.dumps(rows),flush=True)
except Exception as e:
 dump('FAILURE_V1.json',dict(exception_type=type(e).__name__,exception=str(e),traceback=traceback.format_exc()));raise
