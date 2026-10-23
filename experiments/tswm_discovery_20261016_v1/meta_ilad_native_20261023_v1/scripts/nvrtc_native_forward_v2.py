"""Forward-only Python binding for pinned Apache-2.0 gradsim CUDA kernels.

The original device function bodies remain unchanged in the server source cache.
Only unused host headers/wrappers are excluded from the NVRTC compilation unit.
No installation or modification of Torch, driver or shared source is performed.
"""
import ctypes as C,pathlib,hashlib,os,math
import torch

P=C.c_void_p;I=C.c_int;U=C.c_uint;Z=C.c_size_t;S=C.c_char_p

class NVRTCModule:
 def __init__(self,source,expressions):
  torch.cuda.init();self.device=torch.cuda.current_device();torch.empty(1,device='cuda')
  lib=pathlib.Path(torch.__file__).parent/'lib';self.dll_dir=os.add_dll_directory(str(lib))
  self.n=C.CDLL(str(lib/'nvrtc64_120_0.dll'));self.d=C.WinDLL('nvcuda.dll')
  def bind(lib,name,args,rest=I):
   fn=getattr(lib,name);fn.argtypes=args;fn.restype=rest;return fn
  self.ncreate=bind(self.n,'nvrtcCreateProgram',[C.POINTER(P),S,S,I,C.POINTER(S),C.POINTER(S)])
  self.nexpr=bind(self.n,'nvrtcAddNameExpression',[P,S]);self.ncompile=bind(self.n,'nvrtcCompileProgram',[P,I,C.POINTER(S)])
  self.nlogsize=bind(self.n,'nvrtcGetProgramLogSize',[P,C.POINTER(Z)]);self.nlog=bind(self.n,'nvrtcGetProgramLog',[P,P])
  self.nptxsize=bind(self.n,'nvrtcGetPTXSize',[P,C.POINTER(Z)]);self.nptx=bind(self.n,'nvrtcGetPTX',[P,P])
  self.nname=bind(self.n,'nvrtcGetLoweredName',[P,S,C.POINTER(S)]);self.ndestroy=bind(self.n,'nvrtcDestroyProgram',[C.POINTER(P)])
  self.nerr=bind(self.n,'nvrtcGetErrorString',[I],S)
  self.dinit=bind(self.d,'cuInit',[U]);self.dcontext=bind(self.d,'cuCtxGetCurrent',[C.POINTER(P)])
  self.dload=bind(self.d,'cuModuleLoadDataEx',[C.POINTER(P),P,U,P,P]);self.dfunction=bind(self.d,'cuModuleGetFunction',[C.POINTER(P),P,S])
  self.dlaunch=bind(self.d,'cuLaunchKernel',[P,U,U,U,U,U,U,U,P,C.POINTER(P),P])
  self.dunload=bind(self.d,'cuModuleUnload',[P]);self.dgeterr=bind(self.d,'cuGetErrorString',[I,C.POINTER(S)])
  self.dcheck(self.dinit(0));ctx=P();self.dcheck(self.dcontext(C.byref(ctx)));assert ctx.value
  self.context=ctx.value;major,minor=torch.cuda.get_device_capability();self.options=['--std=c++14',f'--gpu-architecture=compute_{major}{minor}']
  pr=P();self.ncheck(self.ncreate(C.byref(pr),source.encode(),b'pinned_forward.cu',0,None,None))
  try:
   for ex in expressions:self.ncheck(self.nexpr(pr,ex.encode()))
   opts=(S*len(self.options))(*(x.encode() for x in self.options));code=self.ncompile(pr,len(opts),opts)
   size=Z();self.ncheck(self.nlogsize(pr,C.byref(size)));log=C.create_string_buffer(size.value);self.ncheck(self.nlog(pr,log));self.compile_log=log.value.decode(errors='replace')
   if code:raise RuntimeError(self.nerr(code).decode()+'\n'+self.compile_log)
   size=Z();self.ncheck(self.nptxsize(pr,C.byref(size)));ptx=C.create_string_buffer(size.value);self.ncheck(self.nptx(pr,ptx));self.ptx=ptx.raw
   names={}
   for ex in expressions:
    p=S();self.ncheck(self.nname(pr,ex.encode(),C.byref(p)));names[ex]=p.value
  finally:self.ndestroy(C.byref(pr))
  self.module=P();self.dcheck(self.dload(C.byref(self.module),C.cast(ptx,P),0,None,None));self.functions={}
  for ex,name in names.items():
   fn=P();self.dcheck(self.dfunction(C.byref(fn),self.module,name));self.functions[ex]=fn
  self.source_sha256=hashlib.sha256(source.encode()).hexdigest();self.ptx_sha256=hashlib.sha256(self.ptx).hexdigest()
 def ncheck(self,code):
  if code:raise RuntimeError(self.nerr(code).decode())
 def dcheck(self,code):
  if code:
   p=S();self.dgeterr(code,C.byref(p));raise RuntimeError(f'CUDA driver {code}: '+str(p.value))
 def launch(self,name,blocks,threads,args):
  assert torch.cuda.current_device()==self.device
  ctx=P();self.dcheck(self.dcontext(C.byref(ctx)));assert ctx.value==self.context
  argv=(P*len(args))(*(C.cast(C.pointer(x),P).value for x in args))
  self.dcheck(self.dlaunch(self.functions[name],blocks,1,1,threads,1,1,0,P(torch.cuda.current_stream().cuda_stream),argv,None))
 def close(self):
  if getattr(self,'module',None) and self.module.value:
   torch.cuda.synchronize();self.dcheck(self.dunload(self.module));self.module=P()

class NativeForward:
 def __init__(self,source_root):
  p=pathlib.Path(source_root)/'gradsim/renderutils/cuda/soft_rasterize_cuda_kernel.cu';s=p.read_text()
  end=s.index('__global__ void backward_soft_rasterize_cuda_kernel');end=s.rfind('template <typename scalar_t>',0,end)
  prefix=s[:end];assert prefix.count('__global__ void ')==2
  unit='\n'.join(x for x in prefix.splitlines() if not x.strip().startswith('#include'))+'\n}\n'
  self.inv='&forward_soft_rasterize_inv_cuda_kernel<float>';self.raster='&forward_soft_rasterize_cuda_kernel<float>'
  self.module=NVRTCModule(unit,[self.inv,self.raster]);self.original_source_sha256=hashlib.sha256(p.read_bytes()).hexdigest()
 def forward(self,faces,textures,faces_info,aggrs_info,soft_colors,image_size,near,far,eps,sigma_val,func_id_dist,dist_eps,gamma_val,func_id_rgb,func_id_alpha,texture_sample_type,double_side):
  tensors=[faces,textures,faces_info,aggrs_info,soft_colors]
  assert all(x.is_cuda and x.dtype==torch.float32 and x.is_contiguous() for x in tensors)
  assert len({x.device for x in tensors})==1
  b,n=faces.shape[:2];t=textures.shape[2];res=int(math.sqrt(t));threads=512
  self.module.launch(self.inv,(b*n+threads-1)//threads,threads,[P(faces.data_ptr()),P(faces_info.data_ptr()),I(b),I(n),I(image_size)])
  args=[P(x.data_ptr()) for x in tensors]+[I(b),I(n),I(image_size),I(t),I(res),C.c_float(near),C.c_float(far),C.c_float(eps),C.c_float(sigma_val),I(func_id_dist),C.c_float(dist_eps),C.c_float(gamma_val),I(func_id_rgb),I(func_id_alpha),I(texture_sample_type),C.c_bool(double_side)]
  self.module.launch(self.raster,(b*image_size*image_size+threads-1)//threads,threads,args)
  return faces_info,aggrs_info,soft_colors
 def backward(self,*args,**kwargs):raise NotImplementedError('Forward-only data-generation binding; no model training supported')
