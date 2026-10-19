# C48 completed real-process memory comparison

UTC 2026-10-17T13:59:36.191934+00:00.16 fresh fits,40750 formal updates;44 raw groups /1694 model-origin-reveal scores from77 chronological origins across2 real processes. Independent CSV/ZIP, windows, model and metric reconstruction PASS. Max model difference 1.49e-06; window and score arithmetic differences0. No hard-cap or final-step selections,0 numerical-refinement flags. Origins are temporally dependent.

| Process | Existing arm | Open-loop48h MSE | After true24h reveal: final24h MSE |
|---|---|---:|---:|
| ett | node | 0.0646841 | 0.0417644 |
| ett | fixed_dde | 0.0679714 | 0.0435282 |
| ett | learned_dde | 0.0675558 | 0.0434121 |
| ett | gru | 0.0658051 | 0.0431851 |
| ett | ridge | 0.0732976 | 0.0452926 |
| ett | persistence | 0.0779690 | 0.0496342 |
| ett | seasonal24 | 0.1123274 | 0.0644178 |
| jena | node | 0.1551004 | 0.1556371 |
| jena | fixed_dde | 0.1550265 | 0.1358621 |
| jena | learned_dde | 0.1575976 | 0.1297083 |
| jena | gru | 0.1382987 | 0.1104120 |
| jena | ridge | 0.1598312 | 0.1188755 |
| jena | persistence | 0.2075381 | 0.1584764 |
| jena | seasonal24 | 0.2477439 | 0.1722848 |

MSE is in TRAIN-standardized units. Neural rows average two optimization seeds; deterministic controls have one fit. The two columns score different horizons and must not be compared as reveal gains. See same_target_reveal.csv for identical-target comparisons.

Real new observations improve the same final24-hour target with ordinary controls: GRU tail MSE ETT0.0832203 ->0.0431851 and Jena0.1992946 ->0.1104120; ridge ETT0.0946744 ->0.0452926 and Jena0.2392916 ->0.1188755. Means are over two GRU seeds and one deterministic ridge fit. These are point-forecast updates, not Bayesian-law consistency proofs.

Deprioritize the current explicit-delay route as a proposed new framework. It provides no stable advantage over the competent history-aware controls. On Jena, GRU is better than learned delay in both seeds for open-loop and updated forecasts. On ETT, learned versus GRU depends on seed; the weak scalar NODE has the lowest mean open-loop MSE among these arms. This does not rule out useful memory generally or establish NODE as a universal world model.

Full per-origin/seed values, paired differences and seed ranges: outputs/analysis_v1/RESULTS.md, aggregate.csv, paired_model_differences.csv and same_target_reveal.csv. Qualification/verification are computation evidence, not novelty evidence. These are two univariate conditional-mean tasks; probability calibration and reusable stochastic-world learning were not tested.
