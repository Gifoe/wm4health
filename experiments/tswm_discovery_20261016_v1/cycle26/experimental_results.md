# Cycle26: forcing-interface falsifier

Decision: response error from this protocol does not establish a new world-model architecture gap. General reusable time-series world modeling remains the goal; no final direction accepted.

This is a source-derived full-state oracle audit, not a learned-model comparison, partial-observation solution, or native-paper reproduction. The native one-Euler-step forcing definition is correct for its discrete source. We test a deliberately coarsened extension. No new training, test-selected setting or architecture claim.

Three fresh Monte Carlo cohorts,8192 independent initial-state/noise realizations each,linear/nonlinear cases of the same triad family,four cadences,four nonzero signed forcing amplitudes,common Brownian increments. These are not8192 different parameterized systems. 864 score rows;288 three-cohort aggregates. Five methods for the linear case,four for nonlinear. All retained.

| Case at h=.64 | Endpoint | Symmetric | Split4 | Exact discrete input |
|---|---:|---:|---:|---:|
| Linear control | 76.4589% | 12.4296% | 15.8646% | 0.0000% |
| Nonlinear source | 75.7138% | 12.7942% | 15.7219% | not applicable |

Values are mean normalized paired path RMSE of x1 over all common observation times,averaged across signed amplitudes and MC cohorts. Individual signs/amplitudes,full-state scores,mean responses and variance responses remain in rows.csv/aggregate.csv. Full-state path/mean RMSE uses all3 coordinates;endpoint means,endpoint relative errors,endpoint SE and variance responses always refer to x1,also in full_state rows. The raw legacy field n_systems_per_mc means stochastic path realizations,not different parameterized plants. They are not cross-dataset generalization or training-seed statistics. Zero autonomous discrepancy is an exact per-path gate,not merely comparable forecast MSE.

The analytic discrete input operator matches the linear forced reference within numerical precision. Symmetric insertion and ordinary subcycling are old numerical controls. This is direct counterevidence to attributing the endpoint artifact to latent-state learning; it does not prove all nonlinear partial-observation response problems solved.

Main runtime 67.20s; peak CUDA allocated 206.94MiB. Runtime includes generation,compression and scoring,not just GPU computation. {'python': '3.10.20', 'torch': '2.8.0+cu128', 'numpy': '1.26.4', 'gpu': 'NVIDIA GeForce RTX 5090'}. Raw arrays server-only;compact outputs mirrored. Source repository36files2,688,628bytes,exact Git blobs verified locally; source inspection only.

Independent verification: {'hash_files': 28, 'replayed_paths': 2700, 'analytic_values': 81100800, 'score_values': 7776}; maxima {'replay_abs': 8.881784197001252e-16, 'linear_response_abs': 1.921802560839181e-15, 'score_abs': 2.2437607327674414e-13}. Main resultsSHA256=bfa53eccf690cc51fcce9dead49e39f33f9dad65c1a25439ed0394f03fed3d0e. NumPy replay,all score recomputation and independent matrix-power response checks passed. Discrete reference only:no continuous-SDE refinement claim. Protocol fixes the native dt=.01 process and exposes all3 coordinates.

[Falasca2025 exact v4](https://arxiv.org/html/2506.22552v4), [Falasca–Zanna2026 exact v5](https://arxiv.org/html/2602.13847v5), [pinned source](https://github.com/FabriFalasca/Physics-and-causally-constrained-neural-models/tree/398082d6dcee2e9d42832dfcd39febfa48c383dd).
