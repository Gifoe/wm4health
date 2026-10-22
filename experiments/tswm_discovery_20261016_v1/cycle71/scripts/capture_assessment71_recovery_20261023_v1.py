"""Call the unchanged frozen evaluator from a recorded, unrestricted parent process."""
import sys,pathlib,subprocess,json,hashlib,datetime
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from common71_v1 import read,sha,dump,OUT,utc,storage
from runtime_affinity71_v1 import configure

STEM='free_simulation71_recovery_20261023_v1'
script=ROOT/'scripts/evaluate_free_simulation71_v3.py'
assert sha(script)=='2f09e76a7630e5f8cc63755001ff442628fa85574140d5b8c90cf915b5fdb0f7'
assert sha(ROOT/'scripts/evaluate_free_simulation71_v1.py')=='a7c0ffab12752976c72c45d06977edef0c4982d96686e703d2f6ba8f15153769'
assert sha(ROOT/'FREE_SIMULATION_PROTOCOL71_V1.md')=='261b702e630543728475a187c13f110b1a97a6bd2ae9af66089db35a1a0299c3'
assert read(OUT/'free_simulation71_v3_exit.json')['exit_code']==1
assert read(OUT/'bounded_campaign71_v1_formal_exit.json')['exit_code']==0
assert not (OUT/'FREE_SIMULATION_ASSESSMENT71_V1.json').exists()
assert not list(storage('artifact').glob('free_sim*.npz'))
affinity=configure('read')
assert len(affinity['available_efficiency_classes'])>=2
record=dict(utc=utc(),status='UNCHANGED_FROZEN_EVALUATOR_RELAUNCH',parent_cpu_context=affinity,
            prior_exit_sha256=sha(OUT/'free_simulation71_v3_exit.json'),prior_failure_log_sha256=sha(OUT/'free_simulation71_v3_stdout.log'),
            evaluator_sha256=sha(script),protocol_sha256=sha(ROOT/'FREE_SIMULATION_PROTOCOL71_V1.md'),
            scientific_code_changed=False,checkpoint_selection_changed=False,training_restarted=False,reserved_eval_rows_accessed=0)
dump(OUT/'ASSESSMENT_EXECUTION_RECOVERY71_20261023_V1.json',record)
log=OUT/(STEM+'_stdout.log')
with log.open('xb') as stream:
    child=subprocess.Popen([sys.executable,'-I','-B','-X','faulthandler','-u',str(script)],stdout=stream,stderr=subprocess.STDOUT,cwd=ROOT)
    dump(OUT/(STEM+'_launch.json'),dict(pid=child.pid,utc=utc(),script=script.name,evaluator_sha256=sha(script),capture_pid=__import__('os').getpid()))
    code=child.wait()
dump(OUT/(STEM+'_exit.json'),dict(pid=child.pid,exit_code=code,utc=utc(),log_sha256=sha(log)))
print(json.dumps(dict(status='FROZEN_ASSESSMENT_TERMINAL',pid=child.pid,exit_code=code)),flush=True)
sys.exit(code)
