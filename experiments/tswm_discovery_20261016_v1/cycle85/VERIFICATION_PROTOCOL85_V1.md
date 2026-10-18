# Independent verification contract

This verifier is authored while the frozen experiment runs; partial results have been seen. It does not change any model, training budget, selector or evaluation metric. Its purpose is independent reconstruction, not a new hypothesis test.

Verify all completed cases and source qualifications. Replay the exact selected frozen checkpoints and both ordinary dictionaries on every saved EVAL input. Prediction tolerance: atol=rtol=2e-5. Recompute trajectory and response scores directly from the stored predictions/references in separate code, absolute tolerance1e-10. Check selected neural history minima, learning-rate selection and the combined validation selector without inspecting test performance for selection. Recompute selected target ridge coefficients by augmented least squares rather than normal equations, atol=rtol=1e-5; also record the maximum absolute discrepancy. Recompute source normalization from stored source data.

Independently integrate each EVAL physical trajectory with DOP853, rtol1e-10/atol1e-12, using the saved piecewise-constant commands and a separately implemented vector field. Compare raw physical states to RK4 references: atol=rtol=1e-4. Verify the observation transform, initial noise and shared action prefix. Record full array/scalar counts and failures. These tolerance choices are numerical checks, not performance gates.

Verify hashes of all source/selected checkpoints, raw data, arrays and selection seals; selected model metadata must precede EVAL artifact creation. A failed check must remain visible and be diagnosed without changing scientific thresholds or reselecting a model on EVAL. No raw payload is copied to local storage.
