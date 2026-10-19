"""Verify next source/runtime without fitting or scoring a scientific model."""
import native_control_runtime
import hashlib,importlib,json,os,pathlib,sys,time
os.environ['NUMBA_NUM_THREADS']='1'
ROOT=pathlib.Path(__file__).resolve().parents[1]
started=time.monotonic();source=ROOT/'upstream/koopman_generator_em'
manifest=json.loads((ROOT/'outputs/koopman_source_manifest.json').read_text())
for row in manifest['files']:
 data=(source/row['path']).read_bytes()
 assert hashlib.sha256(data).hexdigest()==row['sha256']
 assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==row['git_blob']
sys.path.insert(0,str(source))
import numpy as np
import scipy
import numba
native=importlib.import_module('actuated_Koopman_EM_4')
result=dict(status='source_hashes_and_import_passed',repository=manifest['repository'],revision=manifest['revision'],
    n_files=len(manifest['files']),numpy=np.__version__,scipy=scipy.__version__,numba=numba.__version__,
    runtime_s=time.monotonic()-started,scope='source integrity and import only; no training, no native-paper reproduction',
    native_class_exists=hasattr(native,'ActuatedKoopmanEM'),numba_threads=numba.get_num_threads())
assert result['native_class_exists'] and result['numba_threads']==1
(ROOT/'outputs/native_control_import_probe.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result),flush=True)
