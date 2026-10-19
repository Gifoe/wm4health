# Baseline registry — cycle16

| Item | Contract and provenance | Execution status |
|---|---|---|
| Native MTS3 | Official ab971c0ec43df7664c4b727ca69d5b8dd0cab5da;115742 parameters; native model YAML, OT history and6 deterministic calendar conditions;151 history/75 forecast | Seed2021 pilot completed1500 updates; best step1100; not an original MTS3 paper reproduction |
| Persistence | Same origins, conditions and targets; horizon-wise variance from training residuals | Validation-only reference completed |
| Seasonal24 | Repeat the preceding24 observations; training residual variance | Validation-only reference completed |
| Ridge ARX | Fixed alpha1; lags1..24,48,72,96,120,144; six calendar features; train-only innovation variance | Validation-only reference completed; not capacity/compute matched |
| Dense fast covariance, cross-scale dependence suppressed | Same trained A,C,Q, initial belief, mean and decoder as native MTS3 | Full selected-checkpoint diagnostic completed on69 origins; independent closed-form verification passed |
| Dense augmented fast/context covariance | Same as previous control, retaining generated fast/context covariance | Full selected-checkpoint diagnostic completed; no NLL gain; ordinary existing inference, not a proposed method |
| Variational manifold system identification | Official selected source89a35d3e3e95e4ad8cc82dc74e579c30a41181db | Source review only; no native training |
| RAMP/MPNP | Versioned primary papers; RAMP code availability unverified | Literature review only |
| Probabilistic meta-system-ID | Official PuraFilippo/sysid_benchmarks4c20d94ccab51abf4ebf68aa8810f4596381dd41;14 source/notebook files; releasev0.1 asset237784764 | Public90193824-byte checkpoint verified on both hosts; restricted weights-only load and native CPU interface checks passed; no benchmark accuracy reproduction |
| FIM-SDE | Primary2502.19049v3 content-pinned; estimates low-dimensional drift and diffusion | Literature review only; no code/model execution |

Training receives no future raw load/wind forecast columns because their issuance-time provenance is unverified. Future conditions are deterministic timestamps. This differs from original WorldTS covariate inputs; no cross-protocol ranking is permitted. The final20% test segment is not evaluated. Dependency facade and alignment corrections are documented before fitting in FULL_MODEL_PLAN.md and PREVIOUS_SCHEMA_ADDENDUM.md. Earlier artifacts remain unchanged.

Additional capability boundaries and source-protocol differences are recorded in METAID_FIM_BOUNDARY.md. Their public-code/weight availability is not a scientific reproduction result.

The released model has5548418 parameters, context400, query initial observations30, horizon100,12 layers. Source14-file verification, strict state-dict loading, finite positive standard deviations, zero future-label-placeholder effects and zero forbidden future-control-prefix effects are recorded in outputs/metaid_preflight/results.json. This preflight uses synthetic random inputs and establishes an executable interface, not forecasting quality.
