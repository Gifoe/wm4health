"""Compare completed pre-crash epoch prefix with the same-initialization retry."""
from native53 import *
import numbers

out = R / 'outputs/driver_crash_replay_prefix_v1.json'; assert not out.exists()
tag = 'PatchTST__obs__seed3'
old_path = ART / 'runtime_microbatch_v6' / tag / 'curve.json'
new_path = ART / 'runtime_microbatch_v7' / tag / 'curve.json'
old = json.loads(old_path.read_text()); new = json.loads(new_path.read_text())
assert len(old) == 27 and len(new) >= 27
init_path = ART / 'runtime_microbatch_v7' / tag / 'retry_initialization.json'
initial = json.loads(init_path.read_text()); assert initial['status'] == 'EXACT_CONFIG_AND_INITIAL_TENSOR_MATCH_TO_NATIVE_CRASH'
def difference(a, b):
    if isinstance(a, dict):
        assert set(a) == set(b); return max([difference(a[k], b[k]) for k in a] or [0.])
    if isinstance(a, numbers.Number): return abs(float(a) - float(b))
    assert a == b; return 0.
maximum = 0.; rng_matches = True
for a, b in zip(old, new):
    assert a['epoch'] == b['epoch']
    for key in ['train', 'val', 'lr_before_scheduler']: maximum = max(maximum, difference(a[key], b[key]))
    rng_matches &= all(a[key] == b[key] for key in ['cpu_rng_before_train', 'cpu_rng_after_val'])
assert maximum < 1e-6 and rng_matches, (maximum, rng_matches)
dump(out, {'utc': now(), 'status': 'PASS_27_EPOCH_NATIVE_CRASH_PREFIX_REPLAY',
    'epochs_compared': 27, 'maximum_train_val_or_lr_error': maximum, 'all_cpu_rng_hashes_equal': rng_matches,
    'original_initial_bytes_equal': initial['previous_initial_sha256'] == initial['new_initial_sha256'],
    'initialization_receipt_sha256': sha(init_path), 'original_curve_sha256': sha(old_path),
    'new_completed_epochs_observed': len(new), 'future_crop_predictions_read': False,
    'scope': 'Completed recorded source TRAIN/VAL prefix only; does not establish driver root cause or certify later epochs.'})
print(out.read_text(), flush=True)
