# C74: physical data and canonical dynamics controls

Two physical systems and nine original training records are now qualified. These are new ordinary-control executions, not a newly proposed world-model framework. No sensor-shift score or official-test score has been computed. No final research direction is accepted.

## Data and information

EMPS supplies one24841-sample original training record, using motor force vir and position qm. F16 supplies four73728-sample multisine records and four108477-sample swept-sine records, using Force and the official default Acceleration channel0. F16 Fs is400Hz. EMPS has strictly positive released clock increments near1ms; their tiny variation is retained without resampling. Keep raw source units; no unit inference from magnitude.

The two official public archives total150405224 bytes, downloaded with normal HTTPS verification. F16 was downloaded directly on the server. After the server's Google connection timed out, the identical official EMPS ZIP went through transient RAM and verified pinned SFTP; no data file was saved locally. Archive/source/member SHA256, sizes and ZIP CRCs are recorded. No publisher-provided checksum is claimed verified. Only training MAT members were extracted and parsed numerically; official test files remain in the archives. No photographs were extracted.

The bounded prior-ledger review checked136 baseline/result files. It found actual Silverbox scores in C29, so Silverbox was excluded. No EMPS/F16 score mention was found in that check; this is not a universal no-exposure certificate. The current independent corpus must retain that provenance limit.

EMPS was collected under feedback. Here the supplied recorded future motor-force path is an explicit condition of simulation. These scores do not identify the consequences of changing its controller or represent an operational forecast with unknown future actions. F16 uses the supplied measured shaker-force path. We do not merge these different experimental input mechanisms into a causal-effect claim.

## What was fit and scored

Each original TRAIN record is split at80% by time; the following256 samples separate its fit segment from DEV forecast origins. One pooled fixed train-only u/y mean and population SD is fit per physical system. Three float64 recursive ridge-ARX controls use lags4,16,64 and fixed regularizer0.001 times the mean non-intercept Gram diagonal. No future y enters a forecast. Each origin supplies128 history samples and128 future input values; all128 outputs are generated recursively without resets. There are582 complete origins,74496 forecast points per method, across the two systems. These are windows, not582 independent physical systems. The two systems have different physical-duration horizons; no cross-system score is averaged.

| Control | EMPS DEV NRMSE | F16 DEV NRMSE |
|---|---:|---:|
| ridge_arx_lag4 | 0.051000862 | 0.480657700 |
| ridge_arx_lag16 | 0.050615636 | 0.384285702 |
| ridge_arx_lag64 | 0.020693531 | 0.335650939 |
| persistence | 0.085628611 | 1.250291648 |
| train_mean | 0.956668734 | 0.873736932 |

All records and all three lags are retained. DEV selects lag64 as the ordinary competence control in each system. All forecasts are finite. Full per-record raw/normalized metrics, counts, coefficient hashes, raw prediction hashes and all runtime details are in outputs/CANONICAL_ARX74_V2.json. Six ridge fits completed; no neural model was trained here. These are discovery DEV scores under a rolling-window task, not official full-record leaderboard scores or an independent test confirmation.

## Verification and engineering provenance

An independent companion-state matrix implementation reproduces54 first/last-window ARX rollouts to max absolute difference1.0658141036401503e-14. All372480 saved forecast-value rows across the five methods were rechecked against the canonical source arrays; saved target differences are exactlyzero and all reported DEV metrics reconstruct. Persistence/mean predictions and source/artifact hashes also match. The first preparation version failed its overly strict uniform-clock assertion before writing a canonical dataset or fitting a model. Its source and terminal log are preserved. The monotone-clock correction changes no split, forecast horizon, regularization or lag. The remote EMPS acquisition failure is also preserved.

## Research consequence

Both original observation interfaces already admit competent ordinary dynamics controls, especially EMPS. Their scores do not prove that observation/dynamics reuse is solved; no constructed interface change has yet been tested. They do prevent treating a weak uncalibrated predictor as the only comparison. F16's remaining error is not itself a research gap and may reflect linear-model limitations, noise or the selected rollout protocol.

The next experiment must freeze a declared measurement-shift curriculum and compare complete nonlinear calibration/observer, state-space, replay and no-reuse controls. Sensor operators will be explicitly constructed over real physical trajectories, not described as measured hardware replacements. Source/calibration raw segments must be disjoint if the setting excludes paired views; separate-device switches create separate episodes. Original official test outputs stay closed until all rules and method choices are fixed. The original C71 full-budget three-seed campaign and its separate frozen assessment remain unchanged. Overall goal ACTIVE/UNACHIEVED; accepted final directions0.
