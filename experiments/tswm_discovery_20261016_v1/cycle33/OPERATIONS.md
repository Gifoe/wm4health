# C33 execution operations

Cache creation completed with all original-source SHA checks and float32 TRAIN/VAL data. Initial Start-Process launch receipt PID42420 existed, but subsequent authoritative CIM queries showed neither that PID nor the training command, no fit directory and zero-byte output/error logs. No optimization occurred. The initial receipt/logs remain unchanged.

Use the previously verified Win32_Process.Create CIM method, hidden startup and a separate capture supervisor, to detach job ownership from the ending SSH session. This changes process lifetime only; model, budget, split, seed, training source and cache remain fixed. Confirm real worker/status progress before calling the training live. Never start a duplicate based only on a polling timeout.

## Independent verifier chronology preflight

Before any TEST evaluation, a TRAIN-only statistics preflight detected that the independent verifier pivoted unparsed date strings. Lexicographic sorting differed from chronological sorting: NP mean/scale maximum differences were 237.07435268465633/134.0207485254432; PJM 164.8505859907309/68.07712676844676. The locked training/cache implementation already parsed datetime and checked a complete monotonically increasing hourly index; it is unchanged.

Preserve verify_readouts33.py and the v1 postprocess supervisor. Stop only the verified still-waiting postprocess real PID41316; retain its status as postprocess_waiter_v1_superseded.json. No TEST directory existed and no evaluation had run. New verify_readouts33_v2.py explicitly parses datetime before independent pivoting. The second TRAIN-only preflight gives NP mean/scale differences 1.8829382497642655e-13/7.283063041541027e-14, PJM 5.684341886080802e-14/4.618527782440651e-14, within the original unchanged 1e-12 tolerance. This corrects the independent observer, not scientific training or evidence thresholds.

New postprocess_pipeline33_v2.py locks the evaluation, verifier V2, analysis, supplementary provenance closure and plot scripts before opening TEST. It still waits for the same original training job and never starts/restarts training. Source/cache/checkpoint/data provenance equality and fresh TRAIN/VAL feature fixtures are additionally checked in close_provenance33.py before interpretation.
