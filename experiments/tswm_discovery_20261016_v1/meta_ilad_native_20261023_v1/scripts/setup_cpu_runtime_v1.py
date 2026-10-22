"""Isolated documented native CPU dependency set; no shared or OS modifications."""
import pathlib,json,subprocess,sys,os,datetime,hashlib,traceback
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'runtime_cpu_v1';OUT.mkdir(exist_ok=False)
def dump(n,x):
 with (OUT/n).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
try:
 storage=json.loads((ROOT.parent/'STORAGE_LOCATION_V1.json').read_text(encoding='utf-8-sig'));cache=pathlib.Path(storage['cache_root']);runtime=cache/'mi311_cpu_20261023_v1';assert not runtime.exists(),'Existing environment must be inspected, never overwritten';env=os.environ.copy();temp=cache/ROOT.name/'pip_temporary';temp.mkdir(parents=True,exist_ok=True);env.update(TEMP=str(temp),TMP=str(temp),PIP_CACHE_DIR=str(cache/ROOT.name/'pip_cache'),PYTHONNOUSERSITE='1',PYTHONPATH='')
 dump('PLAN_V1.json',dict(python_launcher='py -3.11',runtime_path=str(runtime),package_targets=['torch==2.4.1 CPU','torchdiffeq==0.2.5','numpy==2.0.1','matplotlib==3.10.0','scipy==1.15.2'],shared_environment_modified=False,os_or_driver_modified=False))
 def run(name,args):
  p=OUT/(name+'.log')
  with p.open('xb') as f:r=subprocess.run(args,env=env,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
  dump(name+'_exit.json',dict(exit_code=r.returncode,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),log_sha256=hashlib.sha256(p.read_bytes()).hexdigest()));assert r.returncode==0,(name,r.returncode)
 run('create_venv',['py','-3.11','-m','venv',str(runtime)]);py=runtime/'Scripts/python.exe'
 run('install_torch_cpu',[str(py),'-m','pip','install','--disable-pip-version-check','torch==2.4.1','--index-url','https://download.pytorch.org/whl/cpu'])
 run('install_native_packages',[str(py),'-m','pip','install','--disable-pip-version-check','torchdiffeq==0.2.5','numpy==2.0.1','matplotlib==3.10.0','scipy==1.15.2'])
 code="import sys,json,importlib.metadata as m,numpy as n,torch,torchdiffeq,scipy,matplotlib;assert torch.__version__.split('+')[0]=='2.4.1' and n.__version__=='2.0.1';a=n.arange(6,dtype=n.float32).reshape(2,3);b=torch.from_numpy(a).numpy();assert n.array_equal(a,b);print(json.dumps(dict(python=sys.version,torch=torch.__version__,numpy=n.__version__,torchdiffeq=torchdiffeq.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__,cuda_build=torch.version.cuda,numpy_torch_roundtrip=True,packages={d.metadata['Name']:d.version for d in m.distributions()})))"
 r=subprocess.run([str(py),'-I','-B','-c',code],env=env,capture_output=True,text=True,cwd=ROOT)
 (OUT/'qualification_stdout.log').write_text(r.stdout,encoding='utf-8');(OUT/'qualification_stderr.log').write_text(r.stderr,encoding='utf-8');assert r.returncode==0,('Runtime qualification failed',r.returncode,r.stderr[-1500:]);versions=json.loads(r.stdout);dump('ENVIRONMENT_V1.json',versions)
 dump('READY_V1.json',dict(status='NATIVE_META_ILAD_CPU_RUNTIME_READY',runtime_path=str(runtime),python_path=str(py),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),versions=versions,shared_environment_modified=False,os_or_driver_modified=False));print(json.dumps(dict(status='NATIVE_META_ILAD_CPU_RUNTIME_READY',python=versions['python'],torch=versions['torch'],numpy=versions['numpy'])),flush=True)
except Exception as e:dump('FAILURE_V1.json',dict(exception_type=type(e).__name__,exception=str(e),traceback=traceback.format_exc()));raise
