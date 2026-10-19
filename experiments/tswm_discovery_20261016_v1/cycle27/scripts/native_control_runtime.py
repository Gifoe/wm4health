"""Use already verified task-local native dependencies, never alter environments."""
import runtime_profile
import hashlib,json,os,pathlib,sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
PREVIOUS=ROOT.parent/'cycle26'
os.environ['NUMBA_NUM_THREADS']='1'
os.environ['NUMBA_CACHE_DIR']=str(ROOT/'outputs/numba_cache')
sys.path.insert(0,str(PREVIOUS/'vendor/native_control'))
sys.path.insert(0,str(PREVIOUS/'upstream/koopman_generator_em'))
manifest=json.loads((PREVIOUS/'outputs/koopman_source_manifest.json').read_text())
for row in manifest['files']:
 data=(PREVIOUS/'upstream/koopman_generator_em'/row['path']).read_bytes()
 assert hashlib.sha256(data).hexdigest()==row['sha256']
 assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==row['git_blob']
