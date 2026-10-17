"""Acquire exact native core and isolate two lightweight dependencies on the server."""
import datetime, hashlib, json, os, pathlib, re, ssl, subprocess, sys, time, urllib.error, urllib.request
ROOT=pathlib.Path(__file__).resolve().parents[1]; PROJECT=ROOT.parent; OUT=ROOT/'outputs'
MANIFEST=ROOT/'dcl_source_manifest54_v1.json'; PLAN=json.loads(MANIFEST.read_text())
SOURCE=PROJECT/'upstream/dcl_2cd46d13'; ENV=ROOT/'.venv_dcl'; EXE=ENV/'Scripts/python.exe'
STORAGE=json.loads((PROJECT/'STORAGE_LOCATION_V1.json').read_text(encoding='utf-8-sig'))
CACHE=pathlib.Path(STORAGE['cache_root'])/'cycle54/dcl_source_downloads'; CACHE.mkdir(parents=True,exist_ok=True)
BASE='https://raw.githubusercontent.com/dynamical-inference/dcl/'+PLAN['source_revision']+'/'

def gitsha(b): return hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
def acquire(row):
    rel=row['path']; assert re.fullmatch('[A-Za-z0-9_./-]+',rel) and '..' not in pathlib.PurePosixPath(rel).parts
    target=SOURCE/rel; assert target.resolve().is_relative_to(SOURCE.resolve())
    if target.exists(): b=target.read_bytes(); transport='verified existing file'
    else:
        url=BASE+rel
        try:
            with urllib.request.urlopen(url,timeout=20) as response: b=response.read(row['bytes']+1)
            transport='Python default verified TLS'
        except urllib.error.URLError as exc:
            if not isinstance(exc.reason,ssl.SSLCertVerificationError): raise
            temp=CACHE/(row['git_blob_sha1']+'.raw')
            if not temp.exists():
                ps="Invoke-WebRequest -Uri '"+url+"' -OutFile '"+temp.as_posix()+"' -UseBasicParsing -TimeoutSec 20"
                subprocess.run(['powershell','-NoProfile','-Command',ps],check=True,capture_output=True,text=True,timeout=30)
            b=temp.read_bytes(); transport='unchanged Windows default verified TLS'
        assert len(b)==row['bytes'] and gitsha(b)==row['git_blob_sha1'],rel
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as f: f.write(b)
    assert len(b)==row['bytes'] and gitsha(b)==row['git_blob_sha1'],rel
    return {'path':rel,'bytes':len(b),'git_blob_sha1':gitsha(b),'sha256':hashlib.sha256(b).hexdigest(),'transport':transport}

def main():
    receipt=OUT/'DCL_RUNTIME_PREPARATION54_V1.json'; assert not receipt.exists(); rows=[]; start=time.perf_counter()
    for row in PLAN['selected_files']:
        rows.append(acquire(row))
        if len(rows)%10==0: print(json.dumps({'verified_source_files':len(rows),'planned':len(PLAN['selected_files'])}),flush=True)
    if not (ENV/'pyvenv.cfg').exists(): subprocess.run([sys.executable,'-m','venv','--system-site-packages',str(ENV)],check=True)
    assert EXE.exists()
    env=dict(os.environ,PIP_CACHE_DIR=str(pathlib.Path(STORAGE['cache_root'])/'cycle54/pip_dcl'))
    result=subprocess.run([str(EXE),'-m','pip','install','--index-url','https://pypi.org/simple','--disable-pip-version-check','typeguard==2.13.3','jaxtyping==0.2.38'],env=env,capture_output=True,text=True)
    (OUT/'dcl_dependency_install54_v1.stdout.log').write_text(result.stdout,encoding='utf-8'); (OUT/'dcl_dependency_install54_v1.stderr.log').write_text(result.stderr,encoding='utf-8'); result.check_returncode()
    inspect="import importlib.metadata as m,json,sys,torch;print(json.dumps({'python':sys.executable,'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'versions':{d.metadata['Name']:d.version for d in m.distributions()}}))"
    runtime=json.loads(subprocess.check_output([str(EXE),'-c',inspect],text=True)); assert runtime['torch']=='2.8.0+cu128'
    obj={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'EXACT_NATIVE_DCL_CORE_AND_SEPARATE_LIGHTWEIGHT_VENV_READY','source_revision':PLAN['source_revision'],'source_root':str(SOURCE),'selected_files':rows,'source_manifest_sha256':hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),'runtime':runtime,'runtime_root':str(ENV),'existing_RP_runtime_modified':False,'global_package_install':False,'database_connection':False,'paper_signals_downloaded':False,'formal_fits':0,'runtime_seconds':time.perf_counter()-start,'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}
    receipt.write_text(json.dumps(obj,indent=2),encoding='utf-8'); print(json.dumps({'status':obj['status'],'verified_files':len(rows),'runtime_seconds':obj['runtime_seconds']}),flush=True)
if __name__=='__main__': main()
