# C87 native nonlinear data-selection comparison

This is a controlled native-model adaptation study, not full-paper reproduction. Learned-rollout numerical qualification remains incomplete; see NUMERICAL_OUTCOME87_V1.md. Simulation comparisons below describe the declared RK4/TF32 protocol. Final accepted research directions remain0. Exact released-checkpoint training data are unavailable. Source and native-model qualification succeeded; held-out model selection remained sealed.

## Source competence

| seed | native arm | selected update | common VAL normalized field L1 |
|---|---|---:|---:|
| 2026872 | whole_filter | 2500 | 17.4475 |
| 2026872 | resampled_ic | 2500 | 17.8117 |
| 2026872 | unselected | 3000 | 18.5502 |
| 2026873 | whole_filter | 2500 | 17.8417 |
| 2026873 | resampled_ic | 2500 | 18.2859 |
| 2026873 | unselected | 2500 | 18.6105 |
| 2026874 | whole_filter | 2500 | 17.2709 |
| 2026874 | resampled_ic | 3000 | 17.2047 |
| 2026874 | unselected | 2500 | 17.8148 |

The existing empirical-prior controls use the same source32 independent field labels, without family or parameter labels. Their two-component/noise choices were selected before test opening. This is an ordinary empirical Bayes procedure, not a proposed novel method.

- whole_filter: VAL native-normalized L1 2.73987; raw field MSE 0.20187.
- unselected: VAL native-normalized L1 2.73504; raw field MSE 0.201767.

## Source screening

- train duffing: 7313/8192 retained (89.270%); all numerical paths finite.
- train stuart_landau: 7941/8192 retained (96.936%); all numerical paths finite.
- val duffing: 924/1024 retained (90.234%); all numerical paths finite.
- val stuart_landau: 998/1024 retained (97.461%); all numerical paths finite.

## Held-out results

Each cohort has128 mechanisms per family. Retained/crossed contexts preserve the mechanism/IC-pair marginals but change their joint pairing and visited-state information. New-IC free simulation uses no future observations. Every failure remains counted. Population MSE is undefined if any path fails; finite-only values are explicitly separated in JSON/CSV.

| model | family | cohort | H10 failures | population MSE | population mean NRMSE | penalized mean NRMSE |
|---|---|---|---:|---:|---:|---:|
| native_frozen | duffing | retained | 0 | 10.3595 | 3.11294 | 3.11294 |
| native_frozen | stuart_landau | retained | 0 | 4.58493 | 3.14824 | 3.14824 |
| native_frozen | duffing | crossed | 0 | 61.528 | 3.7101 | 3.7101 |
| native_frozen | stuart_landau | crossed | 0 | 12.4054 | 3.27985 | 3.27985 |
| native_frozen | duffing | unselected | 0 | 4.92139 | 2.70287 | 2.70287 |
| native_frozen | stuart_landau | unselected | 0 | 4.10901 | 3.0511 | 3.0511 |
| native_seed2026872_whole_filter | duffing | retained | 0 | 0.418719 | 0.800085 | 0.800085 |
| native_seed2026872_whole_filter | stuart_landau | retained | 0 | 0.253993 | 0.77253 | 0.77253 |
| native_seed2026872_whole_filter | duffing | crossed | 0 | 0.455054 | 0.800671 | 0.800671 |
| native_seed2026872_whole_filter | stuart_landau | crossed | 0 | 0.295956 | 0.818253 | 0.818253 |
| native_seed2026872_whole_filter | duffing | unselected | 0 | 0.364653 | 0.733126 | 0.733126 |
| native_seed2026872_whole_filter | stuart_landau | unselected | 0 | 0.248486 | 0.785416 | 0.785416 |
| native_seed2026872_resampled_ic | duffing | retained | 0 | 0.460665 | 0.820683 | 0.820683 |
| native_seed2026872_resampled_ic | stuart_landau | retained | 0 | 0.237874 | 0.685701 | 0.685701 |
| native_seed2026872_resampled_ic | duffing | crossed | 0 | 0.473128 | 0.830393 | 0.830393 |
| native_seed2026872_resampled_ic | stuart_landau | crossed | 0 | 0.243173 | 0.70336 | 0.70336 |
| native_seed2026872_resampled_ic | duffing | unselected | 0 | 0.391315 | 0.773439 | 0.773439 |
| native_seed2026872_resampled_ic | stuart_landau | unselected | 0 | 0.241138 | 0.710237 | 0.710237 |
| native_seed2026872_unselected | duffing | retained | 0 | 0.399559 | 0.804119 | 0.804119 |
| native_seed2026872_unselected | stuart_landau | retained | 0 | 0.172449 | 0.678263 | 0.678263 |
| native_seed2026872_unselected | duffing | crossed | 0 | 0.447977 | 0.793325 | 0.793325 |
| native_seed2026872_unselected | stuart_landau | crossed | 0 | 0.314987 | 0.780852 | 0.780852 |
| native_seed2026872_unselected | duffing | unselected | 0 | 0.316586 | 0.699601 | 0.699601 |
| native_seed2026872_unselected | stuart_landau | unselected | 0 | 0.184566 | 0.715626 | 0.715626 |
| native_seed2026873_whole_filter | duffing | retained | 0 | 0.335306 | 0.758191 | 0.758191 |
| native_seed2026873_whole_filter | stuart_landau | retained | 0 | 0.194914 | 0.653778 | 0.653778 |
| native_seed2026873_whole_filter | duffing | crossed | 0 | 0.411713 | 0.783109 | 0.783109 |
| native_seed2026873_whole_filter | stuart_landau | crossed | 0 | 0.236534 | 0.701556 | 0.701556 |
| native_seed2026873_whole_filter | duffing | unselected | 0 | 0.269556 | 0.685459 | 0.685459 |
| native_seed2026873_whole_filter | stuart_landau | unselected | 0 | 0.195306 | 0.671652 | 0.671652 |
| native_seed2026873_resampled_ic | duffing | retained | 0 | 0.384938 | 0.791305 | 0.791305 |
| native_seed2026873_resampled_ic | stuart_landau | retained | 0 | 0.186958 | 0.665125 | 0.665125 |
| native_seed2026873_resampled_ic | duffing | crossed | 0 | 0.434011 | 0.790481 | 0.790481 |
| native_seed2026873_resampled_ic | stuart_landau | crossed | 0 | 0.312433 | 0.718747 | 0.718747 |
| native_seed2026873_resampled_ic | duffing | unselected | 0 | 0.284517 | 0.674952 | 0.674952 |
| native_seed2026873_resampled_ic | stuart_landau | unselected | 0 | 0.185956 | 0.678178 | 0.678178 |
| native_seed2026873_unselected | duffing | retained | 0 | 0.412441 | 0.845623 | 0.845623 |
| native_seed2026873_unselected | stuart_landau | retained | 0 | 0.220332 | 0.715502 | 0.715502 |
| native_seed2026873_unselected | duffing | crossed | 0 | 0.44215 | 0.811759 | 0.811759 |
| native_seed2026873_unselected | stuart_landau | crossed | 0 | 0.293593 | 0.79995 | 0.79995 |
| native_seed2026873_unselected | duffing | unselected | 0 | 0.35666 | 0.757322 | 0.757322 |
| native_seed2026873_unselected | stuart_landau | unselected | 0 | 0.221113 | 0.717978 | 0.717978 |
| native_seed2026874_whole_filter | duffing | retained | 0 | 0.399483 | 0.80446 | 0.80446 |
| native_seed2026874_whole_filter | stuart_landau | retained | 0 | 0.205012 | 0.729918 | 0.729918 |
| native_seed2026874_whole_filter | duffing | crossed | 0 | 0.454295 | 0.79874 | 0.79874 |
| native_seed2026874_whole_filter | stuart_landau | crossed | 0 | 0.664342 | 0.953639 | 0.953639 |
| native_seed2026874_whole_filter | duffing | unselected | 0 | 0.353165 | 0.73942 | 0.73942 |
| native_seed2026874_whole_filter | stuart_landau | unselected | 0 | 0.214985 | 0.758267 | 0.758267 |
| native_seed2026874_resampled_ic | duffing | retained | 0 | 0.343268 | 0.72451 | 0.72451 |
| native_seed2026874_resampled_ic | stuart_landau | retained | 0 | 0.309927 | 0.798492 | 0.798492 |
| native_seed2026874_resampled_ic | duffing | crossed | 0 | 0.417364 | 0.765615 | 0.765615 |
| native_seed2026874_resampled_ic | stuart_landau | crossed | 0 | 0.29229 | 0.807324 | 0.807324 |
| native_seed2026874_resampled_ic | duffing | unselected | 0 | 0.267947 | 0.645859 | 0.645859 |
| native_seed2026874_resampled_ic | stuart_landau | unselected | 0 | 0.309777 | 0.816321 | 0.816321 |
| native_seed2026874_unselected | duffing | retained | 0 | 0.41599 | 0.837015 | 0.837015 |
| native_seed2026874_unselected | stuart_landau | retained | 0 | 0.206053 | 0.662663 | 0.662663 |
| native_seed2026874_unselected | duffing | crossed | 0 | 0.4398 | 0.790872 | 0.790872 |
| native_seed2026874_unselected | stuart_landau | crossed | 0 | 0.254818 | 0.724795 | 0.724795 |
| native_seed2026874_unselected | duffing | unselected | 0 | 0.330342 | 0.724044 | 0.724044 |
| native_seed2026874_unselected | stuart_landau | unselected | 0 | 0.212514 | 0.682861 | 0.682861 |
| ordinary_weak_polynomial | duffing | retained | 3 | undefined | undefined | 31.1523 |
| ordinary_weak_polynomial | stuart_landau | retained | 1 | undefined | undefined | 11.5713 |
| ordinary_weak_polynomial | duffing | crossed | 7 | undefined | undefined | 64.3924 |
| ordinary_weak_polynomial | stuart_landau | crossed | 1 | undefined | undefined | 10.5232 |
| ordinary_weak_polynomial | duffing | unselected | 1 | undefined | undefined | 15.0224 |
| ordinary_weak_polynomial | stuart_landau | unselected | 1 | undefined | undefined | 11.434 |
| persistence | duffing | retained | 0 | 0.762281 | 1.40643 | 1.40643 |
| persistence | stuart_landau | retained | 0 | 1.22221 | 2.00623 | 2.00623 |
| persistence | duffing | crossed | 0 | 0.762281 | 1.40643 | 1.40643 |
| persistence | stuart_landau | crossed | 0 | 1.22221 | 2.00623 | 2.00623 |
| persistence | duffing | unselected | 0 | 0.76233 | 1.43542 | 1.43542 |
| persistence | stuart_landau | unselected | 0 | 1.23444 | 2.06133 | 2.06133 |
| empirical_prior_whole_filter | duffing | retained | 0 | 0.0323186 | 0.16999 | 0.16999 |
| empirical_prior_whole_filter | stuart_landau | retained | 0 | 0.000846704 | 0.0352512 | 0.0352512 |
| empirical_prior_whole_filter | duffing | crossed | 0 | 0.0584614 | 0.215967 | 0.215967 |
| empirical_prior_whole_filter | stuart_landau | crossed | 0 | 0.000444829 | 0.0296324 | 0.0296324 |
| empirical_prior_whole_filter | duffing | unselected | 0 | 0.0555888 | 0.210414 | 0.210414 |
| empirical_prior_whole_filter | stuart_landau | unselected | 0 | 0.00069968 | 0.0347167 | 0.0347167 |
| empirical_prior_unselected | duffing | retained | 0 | 0.0330013 | 0.173386 | 0.173386 |
| empirical_prior_unselected | stuart_landau | retained | 0 | 0.000846726 | 0.0352521 | 0.0352521 |
| empirical_prior_unselected | duffing | crossed | 0 | 0.0585746 | 0.21614 | 0.21614 |
| empirical_prior_unselected | stuart_landau | crossed | 0 | 0.000444728 | 0.0296356 | 0.0296356 |
| empirical_prior_unselected | duffing | unselected | 0 | 0.0559739 | 0.212331 | 0.212331 |
| empirical_prior_unselected | stuart_landau | unselected | 0 | 0.000700139 | 0.0347219 | 0.0347219 |

## Evidence limits and verification

Replayed all63 native source-VAL checkpoints and all reported metric rows. Native field arrays were independently queried through direct model_forward (1274880 scalars); high-accuracy physical truth and empirical Gaussian conditioning were checked. Full-population nonlinear learned rollout integration was not independently replayed. Supplementary learned-rollout checks failed the declared1% criterion on the frozen model and did not reach the adapted-model checks. See NUMERICAL_QUALIFICATION87_V1.json and NUMERICAL_OUTCOME87_V1.md. Metrics are fixed-protocol numerical observations, not certified continuous-time errors. See VERIFICATION87_V4.json for exact scope.

Native source losses and ordinary controls must be considered before assigning a sampler-specific failure. Resampling changes both native observation contexts and its appended supervised field-query locations. A fixed-supervision control is required for shortcut attribution. C87 does not establish novelty, released-checkpoint causality, or a need for a new architecture.
