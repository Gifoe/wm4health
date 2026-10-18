"""Single-use captured worker; fail closed and never retry failed training."""
import sys,pathlib,subprocess
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from common71_v1 import *

def run_request(request):
    name=request['tag'];assert name.replace('_','').isalnum()
    p=OUT/(name+'_request.json');dump(p,request)
    log=OUT/(name+'_stdout.log')
    with log.open('xb') as f:
        child=subprocess.Popen([sys.executable,'-I','-B','-X','faulthandler','-u',str(ROOT/'scripts'/'bounded_native71_v1.py'),'--request',str(p)],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT)
        dump(OUT/(name+'_launch.json'),dict(pid=child.pid,utc=utc(),request_sha256=sha(p),cuda_launch_blocking=os.environ.get('CUDA_LAUNCH_BLOCKING')))
        code=child.wait()
    dump(OUT/(name+'_exit.json'),dict(pid=child.pid,exit_code=code,utc=utc(),log_sha256=sha(log)))
    assert code==0,('Native segment failed; controller stops without retry',name,code)
    result=read(OUT/(name+'_segment_result.json'));assert result['status']=='NATIVE_BOUNDED_SEGMENT_COMPLETE'
    assert sha(result['last']['path'])==result['last']['sha256']
    return result
