"""Persist numerical/probabilistic checks as machine-readable evidence."""
import hashlib,importlib.metadata,io,json,pathlib,platform,time,unittest
ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
 start=time.perf_counter();stream=io.StringIO();suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_*.py');result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
 text=stream.getvalue();print(text,flush=True)
 record={'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'passed':result.wasSuccessful(),'test_output':text,
  'runtime_seconds':time.perf_counter()-start,'python':platform.python_version(),'packages':{p:importlib.metadata.version(p) for p in ['numpy','scipy','torch']},
  'test_sha256':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'tests').glob('test_*.py'))}}
 (ROOT/'outputs/validation_checks.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
 if not result.wasSuccessful():raise SystemExit(1)
if __name__=='__main__':main()
