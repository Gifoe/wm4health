# Numerical verification amendment: canonical checker is verify81_v3.py

This amendment discloses a failed independent check and its resolution. No numerical experiment, candidate selection, fitted model, prediction, context seed, cohort, or declared verification tolerance was changed.

The initial checker used direct trapezoidal integration of `base / state_std` in float64. The experiment's native base vector-field outputs are float32; it first calculates `.5 * (base[1:] + base[:-1])` in float32 and then divides by the float64 state standard deviation and multiplies by time intervals. These two orders are mathematically equivalent but are different floating-point observation operators.

On system35 learned residual, changing the observation operator changed the weak response by at most 2.95e-8 and the independently fitted raw coefficient by 0.0121123. The largest stored coefficient was 31505.46, and the smallest weak-column RMS was 8.14e-7. Direct recomputation with the original normal-equation arithmetic reproduced all24 saved coefficient matrices exactly, isolating this difference from model loading or stochastic features.

The canonical `verify81_v3.py` preserves the original float32 midpoint operation, then independently sums each window and solves augmented least squares instead of using cumulative sums and normal equations. The original absolute/relative coefficient tolerances remain `5e-7` / `5e-6`. It verifies 9312 coefficients; maximum raw coefficient difference is 2.792808118101675e-7. All72 raw trajectory records,288 source-identity arrays, selected candidates and scores pass. One final trajectory fails the declared solver/finite-completeness criterion and remains reported, rather than being dropped.

`outputs/SOLVER_DIAGNOSTIC81_V1.json` preserves both arithmetic comparisons. The failed check is an audit error in matching the finite-precision operator, not evidence that the large new-IC errors were corrected: the original predictions and frozen negative decision are unchanged. `outputs/VERIFICATION81_V1.json` was created only by the passing canonical checker. Earlier checker source versions remain for provenance and must not be mistaken for successful executions.

Administrative export preflight versions only changed snapshot filtering. The final publication snapshot adds this disclosure, the canonical readme and the solver diagnostic; it does not rewrite closed results. No reserved-evaluation system was scored.
