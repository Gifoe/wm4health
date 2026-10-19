# Cycle16 competing hypotheses after full learning

| Hypothesis | Evidence | Judgment |
|---|---|---|
| H1: the native moment approximation affects learned forecasts | Dense controls alter predictive variance; joint average absolute relative change9.635% | Mechanism is active, not a demonstrated predictive failure |
| H2: spatial projection explains a large part of the change | Dense-fast/reset-cross control changes variance49.923% and lowers coverage to90.899% | Supported as a necessary attribution distinction |
| H3: learning/decoder compensation absorbs native approximation | Native has lower selected-checkpoint NLL than either post-hoc control | Plausible, not proven causal explanation; corrected retraining not executed |
| H4: results arise from gross lack of learning | Native NLL0.231094 vs initialized.900980; ordinary references and full curve available | Gross nonlearning not observed; optimal training adequacy unproved |
| H5: a specialized covariance architecture is necessary | Ordinary dense inference already cheap in cycle15; no post-hoc NLL gain here | Unsupported; stop this method line |
| H6: reusable uncertainty/in-context dynamics is itself novel | Source-grounded Bayesian, NP, RAMP, meta-ID and FIM prior art | Rejected as a broad novelty claim |

No true fast/context joint posterior was recovered. Both controls assume the same factorized initial belief. The mean is fixed and the decoder unchanged. No inference from these controls establishes a new trained method or general failure. Final test was not evaluated and no post-hoc hyperparameter search was performed.

Next research must distinguish calibrated marginal forecasts from joint simulation and evidence update capabilities under an explicit contract, while including existing coherent generative/functional models. That is a candidate test boundary, not an accepted contribution. A point forecaster is not required to supply a Bayesian simulator.
