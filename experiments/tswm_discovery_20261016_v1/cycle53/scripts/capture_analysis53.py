"""Wait for locked evaluation verification; analyze/render without restarting fits."""
from native53 import *
import ctypes
from ctypes import wintypes
import subprocess
import time

state = R / 'outputs/analysis_supervisor_v1.json'; assert not state.exists()
qualification = R / 'outputs/analysis_qualification_v1/qualification.json'
q = json.loads(qualification.read_text()); assert q['status'] == 'PASS_PAIRED_RATIO_NULL_DIRECTION_AND_CALENDAR_IDENTITIES'
assert q['analysis_script_sha256'] == sha(R / 'scripts/analyze_native53.py')
assert q['analysis_supplement_sha256'] == sha(R / 'ANALYSIS_SUPPLEMENT.md')
source = R / 'outputs/evaluation_supervisor_v1.json'
started = time.monotonic()
kernel = ctypes.windll.kernel32
kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]; kernel.OpenProcess.restype = wintypes.HANDLE
kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]; kernel.GetExitCodeProcess.restype = wintypes.BOOL
kernel.CloseHandle.argtypes = [wintypes.HANDLE]
def alive(pid):
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle: return False
    value = wintypes.DWORD()
    try:
        assert kernel.GetExitCodeProcess(handle, ctypes.byref(value)); return value.value == 259
    finally: kernel.CloseHandle(handle)
def save(status, **extra): dump(state, {'status': status, 'utc': now(), 'pid': os.getpid(), **extra})
while time.monotonic() - started < 12 * 3600:
    try: parent = json.loads(source.read_text())
    except json.JSONDecodeError: time.sleep(.25); continue
    if parent['status'] == 'NATIVE_TRANSFER_SCORING_AND_ARITHMETIC_VERIFICATION_COMPLETE_ANALYSIS_PENDING': break
    if 'FAILED' in parent['status'] or 'MISSING' in parent['status'] or parent['status'].startswith('WAIT_LIMIT'):
        save('EVALUATION_DEPENDENCY_FAILED_NO_RESTART', dependency=parent); sys.exit(1)
    if not alive(int(parent['pid'])):
        save('EVALUATION_MONITOR_MISSING_NO_RESTART', dependency=parent); sys.exit(2)
    save('WAITING_ON_LIVE_LOCKED_EVALUATION_MONITOR', observed_pid=parent['pid'], observed_status=parent['status'])
    time.sleep(25)
else: save('WAIT_LIMIT_NO_RESTART'); sys.exit(2)
env = os.environ.copy(); env['PYTHONPATH'] = str(P / 'cycle52/runtime_deps') + os.pathsep + env.get('PYTHONPATH', '')
finished = []
for name in ['analyze_native53.py', 'render_native53.py']:
    log = R / 'outputs' / (name + '.log')
    with log.open('wb') as stream:
        child = subprocess.Popen([sys.executable, '-B', '-u', str(R / 'scripts' / name)], cwd=R, env=env, stdout=stream, stderr=subprocess.STDOUT)
        save('RUNNING_POSTVERIFICATION_ANALYSIS', active={'script': name, 'pid': child.pid, 'log': log.name}, finished=finished)
        code = child.wait()
    finished.append({'script': name, 'pid': child.pid, 'exit_code': code})
    if code: save('FAILED_ANALYSIS_NO_FIT_RESTART', finished=finished); sys.exit(1)
save('DESCRIPTIVE_ANALYSIS_AND_FIGURES_COMPLETE_REPORTS_PENDING', finished=finished)
