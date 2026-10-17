"""Continue existing sealed evaluation with its pre-registered time frontier."""
import datetime
import json
import os
import pathlib
import subprocess
import sys
import time

R = pathlib.Path(__file__).resolve().parents[1]
path = R / 'outputs/extension_supervisor_v1.json'
assert not path.exists()
assert (R / 'outputs/COST_RULE_SEAL.json').exists()
started = time.monotonic()
completed = []


def save(status, **extra):
    path.write_text(json.dumps({'status': status, 'pid': os.getpid(), 'completed': completed,
                               'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra}, indent=2), encoding='utf-8')


save('waiting_for_primary_independent_verification')
while time.monotonic() - started < 10800:
    try:
        primary = json.loads((R / 'outputs/evaluation_supervisor_v1.json').read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        time.sleep(1)
        continue
    if primary['status'] == 'FULL_EVALUATION_AND_INDEPENDENT_VERIFICATION_COMPLETE':
        break
    if primary['status'].startswith('FAILED'):
        save('PRIMARY_FAILED_NO_EXTENSION_RUN', error=primary.get('error'))
        sys.exit(1)
    time.sleep(15)
else:
    save('WAIT_LIMIT_REACHED_NO_RESTART')
    sys.exit(2)
for script in ['cost49.py', 'analyze49.py', 'plot49.py']:
    with (R / 'outputs' / (script + '.log')).open('wb') as stream:
        child = subprocess.Popen([sys.executable, '-B', '-u', str(R / 'scripts' / script)], stdout=stream, stderr=subprocess.STDOUT, cwd=R)
        save('running_extension', active_script=script, child_pid=child.pid)
        code = child.wait()
    completed.append({'script': script, 'pid': child.pid, 'exit_code': code})
    if code:
        save('FAILED_EXTENSION', error=script)
        sys.exit(code)
save('TIME_FRONTIER_ANALYSIS_AND_PLOTS_COMPLETE')
