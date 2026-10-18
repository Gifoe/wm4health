# Native runtime retry, before any test scoring

The first campaign sealed four fits, then Informer_PRO seed2022 exited with Windows status3221225477 (0xc0000005). Windows Application event1000 at 2026-10-18 08:36:47 local time identifies VCRUNTIME140.dll in the existing Python environment. This identifies the reported fault module, not its root cause. Available GPU memory after the exit exceeded30GB; no Python OOM exception was recorded. No test scoring had started.

Retain the failed campaign/post states, original stdout and last completed-epoch progress. Retry that same missing fit once using exactly the same seed, source, data, hyperparameters and backend settings; enable Python faulthandler for diagnostics only. Preserve four sealed fits and then execute the four remaining planned fits. Do not change drivers, environments or source models. A repeated native crash requires a new diagnostic decision, not automatic repeated retries.

Successful-fit update totals exclude the aborted attempt. Its last retained progress is epoch3 /960 completed training batches; the exact number of further successful optimizer steps before the native exit is unavailable. Report that uncertainty explicitly. This failure is a runtime incident, not evidence of a dynamical modeling gap.
