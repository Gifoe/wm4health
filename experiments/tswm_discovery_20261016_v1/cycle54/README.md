# C54 jointly learned dynamics audit — comparison running

Snapshot UTC 2026-10-17T18:09:37.216015+00:00: 7/12 formal fits complete. Six corrected ordinary LGSSM EM baselines have completed; the RP-GSSM 10,000-update queue continues. Heldout scoring has not started.

The original native EM baseline produced NaNs because its initial covariance update omitted a sample-count normalization. The standard correction was independently verified before heldout scoring; see [implementation audit](EM_IMPLEMENTATION_AUDIT54.md), [amendment](plan54_em_amendment_v2.json) and [qualification](outputs/QUALIFICATION_EM_FIX54_V1.json). This implementation issue is not a new research gap.

See [protocol](PROTOCOL.md), [reproduction contract](REPRODUCE.md), and [PRO-DYN task-scope audit](PRODYN_SCOPE_AUDIT54.md). For the amended run, use `train54_v3.py` / `run_em54_v2.py` and `postprocess54_v2.py`; the RP training script is unchanged. Complete comparisons and scientific interpretation follow only after all twelve model parameters and training-only readouts are frozen. No new research direction has been accepted. Signals, weights, arrays and failed candidates remain on server storage.
