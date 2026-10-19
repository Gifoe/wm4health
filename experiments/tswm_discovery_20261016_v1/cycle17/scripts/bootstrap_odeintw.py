"""Satisfy native CSTR module's import-only dependency, without changing main env."""
import os, pathlib, subprocess, sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
environment=os.environ.copy();environment['PIP_CONFIG_FILE']=os.devnull
command=[sys.executable,'-m','pip','install','--index-url','https://pypi.org/simple',
    '--only-binary=:all:','--no-deps','--disable-pip-version-check','--retries','1','--timeout','20',
    '--target',str(ROOT/'runtime_dependencies'),'--report',str(ROOT/'outputs/odeintw_install_report.json'),
    'odeintw==1.0.2']
raise SystemExit(subprocess.run(command,env=environment).returncode)
