"""Create a project-local CPU dependency prefix, never change the shared runtime."""
import sys,pathlib,subprocess
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common72_v1 import *
target=storage('cache')/'cpu_dependencies_v1';target.mkdir(exist_ok=False)
log=OUT/'dependency_install72_v1_stdout.log';report=OUT/'dependency_install72_v1_pip_report.json'
packages=['pandas==2.3.3','python-dateutil==2.9.0.post0','pytz==2025.2','tzdata==2025.2','six==1.17.0']
env=dict(os.environ,PIP_CACHE_DIR=str(storage('cache')/'pip_cache'),PIP_DISABLE_PIP_VERSION_CHECK='1')
with log.open('xb') as f:
    result=subprocess.run([sys.executable,'-I','-m','pip','install','--target',str(target),'--no-deps','--only-binary=:all:','--no-input',
        '--disable-pip-version-check','--index-url','https://pypi.org/simple','--report',str(report),*packages],stdout=f,stderr=subprocess.STDOUT,env=env)
assert result.returncode==0,('Scoped dependency install failed',result.returncode)
raw=read(report)
selected=[]
for row in raw['install']:
    from urllib.parse import urlparse
    url=row['download_info']['url'];assert urlparse(url).hostname in ('files.pythonhosted.org','pypi.org')
    selected.append(dict(name=row['metadata']['name'],version=row['metadata']['version'],wheel_sha256=row['download_info']['archive_info']['hashes']['sha256']))
sys.path.insert(0,str(target))
import pandas,numpy,torch
assert pandas.__version__=='2.3.3' and str(pathlib.Path(pandas.__file__).resolve()).startswith(str(target.resolve()))
assert not str(pathlib.Path(numpy.__file__).resolve()).startswith(str(target.resolve()))
assert not str(pathlib.Path(torch.__file__).resolve()).startswith(str(target.resolve()))
obj=dict(utc=utc(),status='SCOPED_CPU_DEPENDENCIES_INSTALLED_WITH_SHARED_NUMPY_TORCH_UNCHANGED',target=str(target),packages=selected,
    pip_report_sha256=sha(report),log_sha256=sha(log),numpy_version=numpy.__version__,torch_version=torch.__version__,
    pandas_version=pandas.__version__,shared_runtime_modified=False,system_configuration_modified=False,
    public_index='https://pypi.org/simple',local_result_files_written=0)
dump(OUT/'DEPENDENCIES72_V1.json',obj)
print(json.dumps({k:obj[k] for k in ['status','packages','numpy_version','torch_version','pandas_version','shared_runtime_modified']}),flush=True)
