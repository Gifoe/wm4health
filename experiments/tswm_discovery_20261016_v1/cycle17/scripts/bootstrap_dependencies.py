"""Install two pinned native-generator dependencies into this cycle only."""
import json, os, pathlib, subprocess, sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
target=ROOT/'runtime_dependencies';assert not target.exists()
environment=os.environ.copy();environment['PIP_CONFIG_FILE']=os.devnull
command=[sys.executable,'-m','pip','install','--index-url','https://pypi.org/simple',
    '--only-binary=:all:','--no-deps','--disable-pip-version-check','--retries','1','--timeout','20',
    '--target',str(target),'--report',str(ROOT/'outputs/dependency_install_report.json'),
    'numba==0.60.0','llvmlite==0.43.0']
raise SystemExit(subprocess.run(command,env=environment).returncode)
