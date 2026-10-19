import runtime_profile
import contextlib,json,os,pathlib,sys,time,traceback
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/dotime';OUT.mkdir(parents=True,exist_ok=True)
sys.path[:0]=[str(ROOT/'vendor'),str(ROOT.parent/'upstream/dotime/src')]
os.environ['HF_ENDPOINT']='https://hf-mirror.com';os.environ['DOTIME_CACHE']=str(ROOT/'data/dotime_cache')
os.environ['HF_HUB_DISABLE_XET']='1'
from inverse_score_audit import Tee
def main():
 from dotime.benchmarks import load_benchmark
 start=time.time();suite=load_benchmark('dot-Identifiability-v1',version='1.2.0',cache_dir=ROOT/'data/dotime_cache')
 print('suite',suite.meta,flush=True)
 from collections import Counter
 counts=Counter(ep.structure for ep in suite)
 print(json.dumps(dict(counts)),flush=True)
 record={'suite':'dot-Identifiability-v1','version':'1.2.0','counts':dict(counts),'episodes':len(suite),'elapsed_seconds':time.time()-start,'native_loader':True,'cache_files':[str(p.relative_to(ROOT)) for p in (ROOT/'data/dotime_cache').rglob('*') if p.is_file()]}
 (OUT/'suite_fetch.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
 print('COMPLETE',flush=True)
if __name__=='__main__':
 with (OUT/'fetch_stdout.log').open('a',encoding='utf-8') as stdout,(OUT/'fetch_stderr.log').open('a',encoding='utf-8') as stderr,contextlib.redirect_stdout(Tee(sys.stdout,stdout)),contextlib.redirect_stderr(Tee(sys.stderr,stderr)):
  try:main()
  except BaseException:traceback.print_exc();raise
