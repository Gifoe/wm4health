"""Wait for existing fits, then run a single locked evaluation and verification."""
import datetime
import json
import os
import pathlib
import subprocess
import sys
import time

R = pathlib.Path(__file__).resolve().parents[1]
path = R / 'outputs/evaluation_supervisor_v1.json'
assert not path.exists()
started = time.monotonic()
completed = []


def save(status, **kwargs):
    path.write_text(json.dumps({'status': status, 'pid': os.getpid(), 'completed': completed,
                               'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), **kwargs}, indent=2), encoding='utf-8')


save('waiting_for_all_fits')
while time.monotonic() - started < 10800:
    supervisor = json.loads((R / 'outputs/supervisor_v1.json').read_text(encoding='utf-8'))
    if supervisor['status'] == 'ALL_EIGHT_FITS_COMPLETE_EVAL_UNOPENED':
        break
    if supervisor['status'].startswith('FAILED'):
        save('FAILED_TRAINING', error=supervisor.get('error'))
        sys.exit(1)
    time.sleep(15)
else:
    save('WAIT_LIMIT_REACHED_NO_RESTART')
    sys.exit(2)
for script in ['score49.py', 'verify49.py']:
    with (R / 'outputs' / (script + '.log')).open('wb') as stream:
        child = subprocess.Popen([sys.executable, '-B', '-u', str(R / 'scripts' / script)], stdout=stream, stderr=subprocess.STDOUT, cwd=R)
        save('running_evaluation', active_script=script, child_pid=child.pid)
        code = child.wait()
    completed.append({'script': script, 'pid': child.pid, 'exit_code': code})
    if code:
        save('FAILED_EVALUATION', error=script)
        sys.exit(code)
save('FULL_EVALUATION_AND_INDEPENDENT_VERIFICATION_COMPLETE')
