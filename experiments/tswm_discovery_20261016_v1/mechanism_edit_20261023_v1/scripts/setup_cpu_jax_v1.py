"""Isolated minimal native-editor runtime; not the full DALI training environment."""
import pathlib,sys,json,subprocess,datetime,hashlib,os
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'runtime_setup_v1';OUT.mkdir(exist_ok=False);ENV=ROOT/'.venv_editor_cpu';assert not ENV.exists()
def record(name,j):
 with (OUT/name).open('x',encoding='utf-8') as f:json.dump(j,f,indent=2)
record('launch.json',dict(pid=os.getpid(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),base_python=sys.executable))
cmd=[sys.executable,'-m','venv',str(ENV)];p=subprocess.run(cmd,capture_output=True);(OUT/'venv_stdout.log').write_bytes(p.stdout+p.stderr);assert p.returncode==0
python=ENV/'Scripts/python.exe';req=['jax==0.4.35','jaxlib==0.4.35','numpy==1.26.4','scipy==1.13.1','ml_dtypes==0.5.1','opt_einsum==3.4.0']
storage=json.loads((ROOT.parent/'STORAGE_LOCATION_V1.json').read_text(encoding='utf-8-sig'));env=dict(os.environ,PIP_CACHE_DIR=str(pathlib.Path(storage['cache_root'])/ROOT.name/'pip'),PIP_DISABLE_PIP_VERSION_CHECK='1')
command=[str(python),'-m','pip','install','--only-binary=:all:','--timeout','30','--retries','2',*req];record('command.json',dict(command=command,packages=req,scope='isolated CPU editor contract only'))
with (OUT/'pip_stdout.log').open('xb') as log:p=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=env)
record('pip_exit.json',dict(exit_code=p.returncode,utc=datetime.datetime.now(datetime.timezone.utc).isoformat()));assert p.returncode==0
p=subprocess.run([str(python),'-I','-c','import jax,jaxlib,numpy,json;print(json.dumps(dict(jax=jax.__version__,jaxlib=jaxlib.__version__,numpy=numpy.__version__,devices=[str(x) for x in jax.devices()])))'],capture_output=True,env=dict(env,JAX_PLATFORMS='cpu'));(OUT/'smoke_stdout.log').write_bytes(p.stdout+p.stderr);assert p.returncode==0
record('RUNTIME_RECEIPT_V1.json',dict(status='ISOLATED_JAX_CPU_READY',smoke=json.loads(p.stdout),packages=req,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),full_dali_dependencies_installed=False,model_executed=False))
print(p.stdout.decode(),flush=True)
