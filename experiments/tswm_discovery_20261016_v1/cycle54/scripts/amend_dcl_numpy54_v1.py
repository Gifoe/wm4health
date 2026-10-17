"""Source-pinned NumPy in the isolated DCL environment only; actual imports recorded."""
import datetime,hashlib,json,os,pathlib,subprocess,sys
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def query():
    code="import numpy,torch,sys,json,importlib.metadata as m;print(json.dumps({'numpy':numpy.__version__,'numpy_file':numpy.__file__,'torch':torch.__version__,'python':sys.executable,'typeguard':m.version('typeguard'),'jaxtyping':m.version('jaxtyping')}))"
    return json.loads(subprocess.check_output([sys.executable,'-c',code],text=True))
def main():
    path=OUT/'DCL_NUMPY_RUNTIME_AMENDMENT54_V1.json';assert not path.exists()
    before=query();assert before['numpy']=='2.2.6';base_file=pathlib.Path(before['numpy_file']);base_hash=sha(base_file)
    storage=json.loads((ROOT.parent/'STORAGE_LOCATION_V1.json').read_text(encoding='utf-8-sig'));env=dict(os.environ,PIP_CACHE_DIR=str(pathlib.Path(storage['cache_root'])/'cycle54/pip_dcl'))
    cmd=[sys.executable,'-m','pip','install','--index-url','https://pypi.org/simple','--disable-pip-version-check','numpy==1.26.4']
    result=subprocess.run(cmd,env=env,text=True,capture_output=True)
    (OUT/'dcl_numpy_install54_v1.stdout.log').write_text(result.stdout,encoding='utf-8');(OUT/'dcl_numpy_install54_v1.stderr.log').write_text(result.stderr,encoding='utf-8');result.check_returncode()
    after=query();assert after['numpy']=='1.26.4' and after['torch']=='2.8.0+cu128' and after['typeguard']=='2.13.3' and after['jaxtyping']=='0.2.38'
    assert pathlib.Path(after['numpy_file']).resolve().is_relative_to((ROOT/'.venv_dcl').resolve());assert sha(base_file)==base_hash
    obj={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'SOURCE_PINNED_NUMPY_AND_ACTUAL_MODULE_PATHS_VERIFIED','before_actual_imports':before,'after_actual_imports':after,'base_numpy_file':str(base_file),'base_numpy_file_sha256_before_and_after':base_hash,'base_numpy_source_file_unchanged':True,'installation_scope':str(ROOT/'.venv_dcl'),'global_install_requested':False,'RP_JAX_runtime_modified':False,'protocol_sha256':sha(ROOT/'DCL_NUMPY_AMENDMENT54_V1.md'),'script_sha256':sha(__file__)}
    path.write_text(json.dumps(obj,indent=2),encoding='utf-8');print(json.dumps(obj),flush=True)
if __name__=='__main__':main()
