"""Add pinned native model-core dependencies to isolated CPU environment only."""
import pathlib,json,subprocess,datetime,os,hashlib
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'runtime_model_core_v1';OUT.mkdir(exist_ok=False);py=ROOT/'.venv_editor_cpu/Scripts/python.exe'
def save(n,j):
 with (OUT/n).open('x',encoding='utf-8') as f:json.dump(j,f,indent=2)
save('launch.json',dict(pid=os.getpid(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
req=['jax==0.4.35','jaxlib==0.4.35','numpy==1.26.4','optax==0.2.3','chex==0.1.87','tensorflow_probability==0.24.0','dm-tree==0.1.8','PyYAML==6.0.2']
storage=json.loads((ROOT.parent/'STORAGE_LOCATION_V1.json').read_text(encoding='utf-8-sig'));env=dict(os.environ,PIP_CACHE_DIR=str(pathlib.Path(storage['cache_root'])/ROOT.name/'pip'),PIP_DISABLE_PIP_VERSION_CHECK='1')
command=[str(py),'-m','pip','install','--only-binary=:all:','--timeout','30','--retries','2',*req];save('command.json',dict(command=command,packages=req,scope='native WorldModel core; no environment/policy stack'))
with (OUT/'stdout.log').open('xb') as f:p=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,env=env)
save('pip_exit.json',dict(exit_code=p.returncode,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),stdout_sha256=hashlib.sha256((OUT/'stdout.log').read_bytes()).hexdigest()));assert p.returncode==0
p=subprocess.run([str(py),'-I','-c','import jax,jaxlib,numpy,optax,tensorflow_probability,json;print(json.dumps(dict(jax=jax.__version__,jaxlib=jaxlib.__version__,numpy=numpy.__version__,optax=optax.__version__,tfp=tensorflow_probability.__version__)))'],capture_output=True,env=dict(env,JAX_PLATFORMS='cpu'));(OUT/'smoke.log').write_bytes(p.stdout+p.stderr);assert p.returncode==0;versions=json.loads(p.stdout);assert versions['jax']=='0.4.35' and versions['numpy']=='1.26.4'
save('READY_V1.json',dict(status='NATIVE_MODEL_CORE_CPU_DEPENDENCIES_READY',versions=versions,full_environment_or_policy_dependencies_installed=False,model_executed=False));print(json.dumps(versions),flush=True)
