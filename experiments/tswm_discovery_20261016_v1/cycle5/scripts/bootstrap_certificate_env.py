"""Create an isolated numerical environment; never modify previous environments."""
import os,pathlib,subprocess,sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
target=ROOT/'.venv_certificate'
python=target/'Scripts'/'python.exe'
if not python.exists():subprocess.run([sys.executable,'-m','venv','--without-pip',str(target)],check=True)
subprocess.run([sys.executable,'-m','pip','--python',str(python),'install','--disable-pip-version-check','--quiet','numpy==1.26.4','scipy==1.15.3','cvxpy==1.6.7','clarabel==0.11.1','scs==3.2.8'],check=True)
subprocess.run([str(python),'-B','-c','import sys,numpy,scipy,cvxpy; print(sys.executable, numpy.__version__, scipy.__version__, cvxpy.__version__, cvxpy.installed_solvers())'],check=True)
