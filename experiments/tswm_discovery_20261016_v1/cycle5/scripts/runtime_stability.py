"""Bounded native-library diagnostic after confirmed Windows access violations."""
import ctypes,hashlib,json,os,pathlib,sys,time
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
if os.name=='nt':
 fn=ctypes.windll.kernel32.SetProcessAffinityMask;fn.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
 assert fn(ctypes.c_void_p(-1),ctypes.c_size_t(0xffff0000))
import numpy as np
import torch
ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
 torch.set_num_threads(1)
 out=ROOT/'outputs'/'runtime_diagnostic';out.mkdir(parents=True,exist_ok=True)
 started=time.perf_counter();iterations=0
 a=np.ones((256,256));b=torch.ones(256,256);c=torch.ones(256,256,device='cuda')
 while time.perf_counter()-started<45:
  r=a@a;s=b@b;t=c@c
  assert np.array_equal(r,np.full_like(r,256.)) and torch.equal(s,torch.full_like(s,256.)) and torch.equal(t,torch.full_like(t,256.))
  iterations+=1
  if iterations%1000==0:print('Arithmetic checks',iterations,flush=True)
 result={'state':'complete','seconds':time.perf_counter()-started,'iterations':iterations,'profile':{'cpu_threads':1,'affinity_mask':'0xffff0000','mkl_threading':'SEQUENTIAL'},'python':sys.version,'executable':sys.executable,'numpy':np.__version__,'numpy_file':np.__file__,'torch':torch.__version__,'torch_file':torch.__file__,'gpu':torch.cuda.get_device_name(0),'claim':'Bounded diagnostic passed, not proof of hardware stability. Earlier unpinned application access violations remain recorded.'}
 (out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()
