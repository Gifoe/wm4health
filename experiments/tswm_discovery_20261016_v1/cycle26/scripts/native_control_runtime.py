"""Process-local dependencies for the pinned native CPU control baseline."""
import runtime_profile
import os,pathlib,sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
os.environ['NUMBA_NUM_THREADS']='1'
os.environ['NUMBA_CACHE_DIR']=str(ROOT/'outputs/numba_cache')
sys.path.insert(0,str(ROOT/'vendor/native_control'))
sys.path.insert(0,str(ROOT/'upstream/koopman_generator_em'))
