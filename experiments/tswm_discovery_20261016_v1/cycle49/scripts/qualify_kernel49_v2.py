"""No-data/no-training timing and equivalence for verification implementation only."""
import datetime
import hashlib
import json
import pathlib
import sys
import time

R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R.parent / 'cycle29/scripts'))
import runtime_profile
import numpy as np
from independent49 import observed_score as original
from independent49_v2 import observed_score as optimized

rng = np.random.default_rng(20265002)
records = []
for h in [12, 24]:
    path = rng.normal(size=(8, 128, h, 2))
    count = rng.poisson(.8, size=(8, h)).astype('float64')
    sums = rng.normal(size=(8, h, 2)) * count[..., None]
    density = np.array([.25, .5, 1., 2., 4., 1., .5, 2.])
    alpha, omega = rng.normal(size=(512, h)) / np.sqrt(h), rng.normal(size=(512, h, 2)) / np.sqrt(h)
    begin = time.perf_counter(); left = original(path, count, sums, density, alpha, omega); old_time = time.perf_counter() - begin
    begin = time.perf_counter(); right = optimized(path, count, sums, density, alpha, omega); new_time = time.perf_counter() - begin
    error = float(np.max(abs(left - right)))
    assert error < 1e-10
    records.append({'horizon': h, 'original_seconds': old_time, 'optimized_seconds': new_time, 'max_absolute_score_difference': error})
out = R / 'outputs/verification_kernel_qualification_v2.json'
assert not out.exists()
result = {'status': 'PASS_NO_TRAINING_VERIFICATION_KERNEL_EQUIVALENCE', 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'records': records, 'formal_weights_or_eval_data_used': False, 'optimization_updates': 0,
          'scope': 'Only independent-verifier kernel arithmetic implementation; no method efficiency claim or tolerance change',
          'source_sha256': hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
          'optimized_source_sha256': hashlib.sha256((R / 'scripts/independent49_v2.py').read_bytes()).hexdigest()}
out.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result), flush=True)
