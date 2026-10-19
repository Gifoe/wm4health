# Frozen native meta-system-ID replication and ordinary-dependence falsification gate

Research scope: reusable time-series world models across unseen systems. This is a baseline/negative-control gate, not an accepted new method. Cycle16 remains sealed and unchanged. No real-data test set, new training, or source modification is involved.

## Native contract

Use PuraFilippo/sysid_benchmarks revision 4c20d94ccab51abf4ebf68aa8810f4596381dd41 and the original released 5,548,418-parameter checkpoint, SHA256 e8324becad85ac610ec94f8ae09048938b343761a5bcd3967a86353a5bf28197. Restricted weights-only deserialization, exact native state dict, non-cached TSTransformer. Original WHDataset generates new Wiener–Hammerstein systems, orders 1–5, native pole ranges, 0.01 output noise. Sequence 930 = context400 + unobserved skip400 + query initial30 + forecast100. Native generator normalizes its intermediate and final trajectories using whole generated sequences, including future outputs. This is an explicitly oracle-normalized *native reproduction protocol*, not prospective raw-unit deployment and not causal intervention evidence.

Calibration:256 systems, generator seeds (system,input,noise)=(20211001,20211002,20211003). Held-out probe:256 systems with seeds (20212001,20212002,20212003). New systems are generated on every draw; units of analysis are systems, not correlated timepoints. No model/threshold selection on the probe. FP32, eval mode, no gradient, batch16, TF32 disabled. Save all generated u/y and predictions. Strictly future y placeholders are zeroed; score independently after inference. Native future-prefix and placeholder invariance already passed cycle16 and are asserted once on this data.

Point controls: last-observation persistence, zero (native normalized mean), and the same released model with context reassigned to a different system (roll within each batch). The context control tests use of system evidence, not causal identification or a competitive forecasting architecture.

## Prespecified dependence controls

Fit only on calibration standardized residuals z=(y-mu)/sigma. Compare native independent Gaussian; global scalar variance temperature E[z²]; Ledoit–Wolf centered full temporal covariance with calibration mean correction; and its diagonal-only counterpart. The last two share the exact same corrected means and marginal variances; their difference isolates ordinary cross-horizon covariance. No new neural model or novel covariance mechanism is proposed.

Report per-system mean marginal NLL, RMSE, 95% marginal coverage; joint NLL per horizon; 95% coverage and scalar Gaussian NLL of trajectory functionals (constant, alternating, first-half minus second-half, and fixed DCT frequencies1–8). All functionals have unit Euclidean norm. Correlated Gaussian joint NLL and functionals are exact for the candidate predictive distributions, not claims that WH residuals are actually Gaussian. Fit/calibration and probe statistics are separate. Monte Carlo bootstrap resamples systems (seed20213001,2000 draws); descriptive pilot intervals do not establish cross-family generality.

## Decisions

Poor native marginal accuracy => do not attribute a stronger joint failure before resolving native reproduction/protocol issues. If ordinary full covariance fixes trajectory summaries without a residual mechanism, close the proposed generic dependence gap; this is old Gaussian calibration. If not, inspect bias, input-conditional errors, generator normalization and existing functional/generative meta-dynamics work before designing any method. Independent-vs-joint density is not itself a novel finding. No posterior requirement is imposed on point-only WorldTS/FIM interfaces.

Nearest prior art already rules out generic novelty: neural processes and Gaussian processes provide function dependence; AR-CNP (ICLR2023) adds correlated predictions by autoregressive deployment; incTNP (ICML2026) provides efficient evidence updates and measures implicit Bayesianness; 2604.13366 studies diffusion in-context robot dynamics. Any remaining candidate must go beyond these.

Overall objective remains active; accepted novel directions0. Data from this gate are exploratory held-out probe data, never a new final benchmark test set for tuning a later method.

Implementation preflight addendum (before any generated data or inference): first captured attempt exited because main environment has no numba. Preserve its complete logs and script snapshot. Install numba0.60.0 + llvmlite0.43.0 into cycle17/runtime_dependencies only, no main environment changes. Redirect Numba JIT cache into this cycle. Retry output gets a fresh directory; all scientific seeds, metrics, controls and sample counts stay as above.

Second import preflight found odeintw absent; it is imported by native CSTR code although the WH experiment does not call it. Add pinned odeintw1.0.2 to the same isolated dependency folder. Both failures occurred before generating samples. Preserve both attempts; scientific protocol unchanged.
