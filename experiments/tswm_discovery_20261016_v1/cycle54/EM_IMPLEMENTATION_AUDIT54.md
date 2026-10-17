# Ordinary EM implementation repair

The two initial native Dynamax 0.1.5 seed0 attempts produced a finite first likelihood and NaNs from the next iteration. They are failed attempts, excluded from the selected comparison. The original wrapper combined finite/monotonic checks on one line; inspection of retained arrays established that finiteness was the failing condition.

The native M-step computes an initial covariance from sums across sequences. Its mean outer product is missing a sample-count normalization. For N sequences, the standard expected covariance is

`m = sum(E[z0]) / N`

`S = sum(E[z0 z0^T]) / N - m m^T`.

In the first training update, the native expression produced a minimum eigenvalue below zero. The corrected expression was positive definite and agreed with an independent NumPy Kalman/RTS expected-moment calculation. All other M-step parameters were identical. The exact numbers, source-function hash and qualification scope are in `outputs/QUALIFICATION_EM_FIX54_V1.json`.

The local subclass calls the native M-step and replaces only this covariance. It retains the native E-step, transition/emission regression updates and fit loop. No installed file, upstream source, data, initialization, seed, iteration budget or RP model was changed. The correction was qualified and declared in `plan54_em_amendment_v2.json` before state readout fitting or heldout evaluation. Two failed 200-iteration native attempts, their NaN arrays and all error logs remain on the server.

The active baseline entrypoint is `train54_v3.py` through `run_em54_v2.py`; the active automatic postprocessor is `postprocess54_v2.py`, using `evaluate54_v2.py` and `analyze54_v2.py`. Earlier source versions and logs are retained for auditability. The final comparison must identify the baseline as ordinary corrected LGSSM EM, not unmodified native EM.

This is a routine implementation repair. It cannot support a claim that ordinary state-space modeling lacks the relevant capability, and it is not a new world-model contribution. [Native implementation reference](https://github.com/probml/dynamax/blob/main/dynamax/linear_gaussian_ssm/models.py).
