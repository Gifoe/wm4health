# C79 runtime amendment, before action-edit measurements

V1 stopped on the first factual CPU qualification, before any suffix/impulse measurement: CrossLinear obs seed0 source-VAL maximum GPU-stored versus CPU-forward difference 5.835294723510742e-5, exceeding the frozen elementwise absolute/relative2e-5 test. This is a failed runtime reproduction check, not an action-response or model failure. Preserve V1 code, pin, fixtures, log and nonzero exit. No arm is replaced and no tolerance is relaxed.

V2 runs all40 unchanged checkpoints and both retained banks on the original CUDA float32 runtime instead of CPU. Keep process-local two CPU threads/efficient-core affinity, batch128, original native forward and all edit definitions. Factual qualification against C53 must still pass before edits. Every model/cell is retained, with no selection based on sensitivity. No new training. Reusing existing VAL/crop identities remains discovery evidence, not independent held-out confirmation. CUDA is already occupied by the independent C71 fixed-budget fit; do not modify that fit or its environment.

The launch-only Start-Process attempt returned a PID but produced no log/outputs; it is not counted as a completed model process. Foreground retained transport then produced the V1 qualification failure above. Subsequent capture uses a retained SSH transport and separate V2 log/exit filenames.
