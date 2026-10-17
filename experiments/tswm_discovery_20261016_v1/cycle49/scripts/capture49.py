"""Single-device sequential eight-fit supervisor; never starts evaluators."""
import datetime
import json
import os
import pathlib
import subprocess
import sys

R = pathlib.Path(__file__).resolve().parents[1]
state = R / 'outputs/supervisor_v1.json'
assert not state.exists()
completed = []
started = datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(status, **extra):
    state.write_text(json.dumps({'status': status, 'pid': os.getpid(), 'started_utc': started,
                                 'completed': completed, **extra}, indent=2), encoding='utf-8')


save('running')
for seed in [2026491, 2026492]:
    for case in ['rotating', 'bistable']:
        for arm in ['rb_score', 'full_clock_elbo']:
            tag = f'{case}__{arm}__seed{seed}'
            logfile = R / 'outputs' / ('train_' + tag + '.log')
            with logfile.open('wb') as stream:
                child = subprocess.Popen([sys.executable, '-B', '-u', str(R / 'scripts/train49.py'), '--case', case, '--arm', arm, '--seed', str(seed)], stdout=stream, stderr=subprocess.STDOUT, cwd=R)
                save('running', active={'tag': tag, 'pid': child.pid, 'log': logfile.name})
                code = child.wait()
            completed.append({'tag': tag, 'pid': child.pid, 'exit_code': code, 'log': logfile.name})
            if code:
                save('FAILED_FIT', error=tag)
                sys.exit(code)
            save('running')
save('ALL_EIGHT_FITS_COMPLETE_EVAL_UNOPENED')
