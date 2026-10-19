import ctypes,os,pathlib,runpy
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
os.environ.update(MKL_THREADING_LAYER='SEQUENTIAL',TSWM_CPU_THREADS='1',TSWM_KUMA_OUT='kuma_native_pems03_affinity1')
fn=ctypes.windll.kernel32.SetProcessAffinityMask;fn.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
assert fn(ctypes.c_void_p(-1),ctypes.c_size_t(0xffff0000))
runpy.run_path(str(pathlib.Path(__file__).with_name('run_kuma_native.py')),run_name='__main__')
