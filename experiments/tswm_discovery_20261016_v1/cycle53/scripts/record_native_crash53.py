"""Preserve a verified native crash and validate completed fits before one retry."""
from native53 import *
import ctypes
from ctypes import wintypes

receipt = R / 'outputs/NATIVE_CRASH_RECEIPT_V1.json'; assert not receipt.exists()
old = json.loads((R / 'outputs/supervisor_v6.json').read_text())
assert old['status'] == 'FAILED_NATIVE_JOB' and old['error'] == 'PatchTST__obs__seed3'
assert old['exit_code'] == 3221225477 and old['native_terminal_files_valid'] is False
event_path = R / 'outputs/native_crash_event_v1.json'
event = json.loads(event_path.read_text()); assert event['module'] == 'nvcuda64.dll' and event['exception_code'] == 'c0000005'
failed = ART / 'runtime_microbatch_v6/PatchTST__obs__seed3'
state = json.loads((failed / 'state.json').read_text())
assert int(event['process_id'], 16) == state['pid']
kernel = ctypes.windll.kernel32
kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]; kernel.OpenProcess.restype = wintypes.HANDLE
kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]; kernel.GetExitCodeProcess.restype = wintypes.BOOL
kernel.CloseHandle.argtypes = [wintypes.HANDLE]
for pid in [old['pid'], state['pid']]:
    handle = kernel.OpenProcess(0x1000, False, int(pid))
    if handle:
        value = wintypes.DWORD()
        try: assert kernel.GetExitCodeProcess(handle, ctypes.byref(value)) and value.value != 259
        finally: kernel.CloseHandle(handle)
assert not (failed / 'summary.json').exists()
assert not (CKPT / 'runtime_microbatch_v6/PatchTST__obs__seed3.pt').exists()
assert not (ART / 'evaluation_v1').exists()
assert not (R / 'outputs/SELECTION_SEAL_V6.json').exists()
curve = json.loads((failed / 'curve.json').read_text()); assert len(curve) == 27
completed = []
for item in old['completed']:
    summary_path = pathlib.Path(item['summary_path']); summary = json.loads(summary_path.read_text())
    terminal = json.loads((summary_path.parent / 'state.json').read_text())
    assert terminal['status'] == 'COMPLETE' and terminal['tag'] == item['tag']
    assert sha(summary['checkpoint']) == summary['checkpoint_sha256']
    completed.append({'tag': item['tag'], 'checkpoint_sha256': summary['checkpoint_sha256'], 'summary_sha256': sha(summary_path)})
assert len(completed) == 29
dump(receipt, {'utc': now(), 'status': 'CONFIRMED_NATIVE_DRIVER_MODULE_ACCESS_VIOLATION',
    'failed_tag': old['error'], 'process_exit_code': old['exit_code'], 'windows_exception': '0xc0000005',
    'event': event, 'root_cause': 'not established; event identifies fault module, not why it faulted',
    'completed_predictive_fits_validated': 28, 'shared_codec_validated': True, 'completed': completed,
    'failed_attempt_completed_recorded_epochs': len(curve), 'failed_attempt_at_least_completed_optimizer_updates': len(curve) * 6,
    'unobserved_partial_updates_possible': True, 'failed_attempt_checkpoint_exists': False,
    'failed_initial_sha256': sha(failed / 'initial_state.pt'), 'failed_config_sha256': sha(failed / 'config.json'),
    'failed_curve_sha256': sha(failed / 'curve.json'), 'failed_log_sha256': sha(R / 'outputs/native_v6_PatchTST__obs__seed3.log'),
    'native_event_sha256': sha(event_path), 'failed_supervisor_sha256': sha(R / 'outputs/supervisor_v6.json'),
    'retry': 'one fresh same-seed/config fit in a new V7 folder, exact initial-tensor/config check before updates; original artifacts retained; no optimizer/checkpoint resume',
    'source_files_rechecked': check_source(), 'transfer_evaluation_opened': False})
print(json.dumps({'status': 'CONFIRMED_NATIVE_DRIVER_MODULE_ACCESS_VIOLATION', 'completed_predictive_fits_validated': 28, 'recorded_failed_epochs': len(curve)}), flush=True)
