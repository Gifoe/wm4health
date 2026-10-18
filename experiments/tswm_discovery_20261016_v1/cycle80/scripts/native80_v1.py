"""Pinned source, inference environment and native loader; no numerical substitutions."""
import sys,pathlib
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common80_v1 import *
MANIFEST=read(OUT/'SOURCE_MANIFEST80_V1.json');SOURCE=pathlib.Path(MANIFEST['source_root'])
DEPS=read(OUT/'DEPENDENCY_INSTALL80_V2.json');assert DEPS['exit_code']==0
sys.path.insert(0,DEPS['target']);sys.path.insert(0,str(SOURCE/'src'));sys.path.insert(0,str(SOURCE/'tutorial/06_ode'))
def check_source():
 for item in MANIFEST['files']:assert sha(SOURCE/item['path'])==item['sha256'],item['path']
 return len(MANIFEST['files'])
def assets():
 m=read(OUT/'ASSET_MANIFEST80_V1.json')
 for a in m['assets']:assert pathlib.Path(a['path']).stat().st_size==a['bytes'] and sha(a['path'])==a['sha256']
 return m['assets']
