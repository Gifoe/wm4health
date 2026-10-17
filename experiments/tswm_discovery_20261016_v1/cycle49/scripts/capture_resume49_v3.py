"""Resume only interrupted verification; retain every existing fit/score byte."""
import datetime
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time

R = pathlib.Path(__file__).resolve().parents[1]
state = R / 'outputs/evaluation_supervisor_v3.json'
assert not state.exists()
receipt = json.loads((R / 'outputs/VERIFIER_PARALLEL_REPAIR_RECEIPT.json').read_text(encoding='utf-8-sig'))
assert receipt['training_and_score_restarted'] is False
assert hashlib.sha256((R / 'outputs/eval_v1/summary.json').read_bytes()).hexdigest() == receipt['completed_primary_summary_sha256']
assert hashlib.sha256((R / 'scripts/verify49_v3.py').read_bytes()).hexdigest() == receipt['replacement_verifier_sha256']
assert not (R / 'outputs/eval_v1/verification.json').exists()
assert not (R / 'outputs/cost_frontier_v1').exists()
assert not (R / 'outputs/analysis_v1').exists()
for i in range(60):
    previous = json.loads((R / 'outputs/evaluation_supervisor_v2.json').read_text(encoding='utf-8'))
    if previous['status'] == 'FAILED_RESUMED_EVALUATION':
        assert previous['error'] == 'verify49_v2.py'
        break
    time.sleep(1)
else:
    raise RuntimeError('Old verifier supervisor not terminal; inspect actual handles')
completed = []


def save(status, **extra):
    state.write_text(json.dumps({'status': status, 'pid': os.getpid(), 'completed': completed,
                                'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                'primary_fit_and_score_restarted': False, **extra}, indent=2), encoding='utf-8')


save('resuming_equivalent_parallel_independent_verification')
for script in ['verify49_v3.py', 'cost49.py', 'analyze49.py', 'plot49.py']:
    with (R / 'outputs' / (script + '.log')).open('wb') as stream:
        child = subprocess.Popen([sys.executable, '-B', '-u', str(R / 'scripts' / script)], stdout=stream, stderr=subprocess.STDOUT, cwd=R)
        save('running', active_script=script, child_pid=child.pid)
        code = child.wait()
    completed.append({'script': script, 'pid': child.pid, 'exit_code': code})
    if code:
        save('FAILED_RESUMED_EVALUATION', error=script)
        sys.exit(code)
save('VERIFICATION_TIME_FRONTIER_ANALYSIS_AND_PLOTS_COMPLETE')
