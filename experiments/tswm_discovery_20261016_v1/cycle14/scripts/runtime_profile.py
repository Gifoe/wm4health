"""Process-local profile that passed cycle5; leaves server defaults untouched."""
import ctypes,os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
os.environ.update(MKL_THREADING_LAYER='SEQUENTIAL',WANDB_MODE='disabled',WANDB_DISABLED='true',WANDB_SILENT='true')
if os.name=='nt':
 fn=ctypes.windll.kernel32.SetProcessAffinityMask;fn.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
 assert fn(ctypes.c_void_p(-1),ctypes.c_size_t(0xffff0000))
