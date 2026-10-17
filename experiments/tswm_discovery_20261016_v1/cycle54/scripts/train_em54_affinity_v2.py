"""Corrected ordinary EM, same protocol, separate eight-CPU execution group."""
import ctypes,pathlib,runpy,sys
import common54
fn=ctypes.windll.kernel32.SetProcessAffinityMask;fn.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
assert fn(ctypes.c_void_p(-1),ctypes.c_size_t(0xff000000))
root=pathlib.Path(__file__).resolve().parents[1];assert sys.argv[1]=='em'
runpy.run_path(str(root/'scripts/train54_v3.py'),run_name='__main__')
