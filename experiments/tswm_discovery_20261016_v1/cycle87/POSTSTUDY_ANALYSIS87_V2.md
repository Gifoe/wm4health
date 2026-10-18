# Numerical qualification update for the paired analysis

These are descriptive comparisons under the fixed RK4/TF32 protocol. Supplementary independent learned-rollout checks did not pass1% on the frozen retained-cohort example; adapted-model checks were not reached. Do not interpret54/54 as a certified continuous-time comparison or a general theorem. See NUMERICAL_OUTCOME87_V1.md. The scoped method route is deprioritized for lack of sufficient novelty/necessity evidence; every possible trajectory-selection effect is not falsified.

# Closed paired effects and ordinary-control decision

Matched-source empirical-prior H10 failure-aware NRMSE is lower in 54/54 native comparisons. This is descriptive within the two bounded cubic families, not a universal statement about all world models.

| family | cohort | B-minus-A mean NRMSE | exploratory conditional-pool interval |3 optimizer-seed differences |
|---|---|---:|---|---|
| duffing | retained | -0.00874622 | [-0.059723624837242646, 0.04218571383690617] | [0.0205977668971833, 0.03311396534048764, -0.07995039350958369] |
| duffing | crossed | 0.00132312 | [-0.05034615939738286, 0.05320755338683865] | [0.029722351453040384, 0.007372547479059164, -0.033125539997407076] |
| duffing | unselected | -0.0212516 | [-0.06610165260553104, 0.02671793304170835] | [0.04031328663314619, -0.010507078713110237, -0.09356086355124062] |
| stuart_landau | retained | -0.00230246 | [-0.041263453681482126, 0.03466526824835487] | [-0.08682879921148101, 0.01134717135732567, 0.06857425118828168] |
| stuart_landau | crossed | -0.0813388 | [-0.2211480122167163, 0.016779831767333463] | [-0.11489291198023388, 0.017191791220330817, -0.14631525798109746] |
| stuart_landau | unselected | -0.00353314 | [-0.03705487272696578, 0.030562344918343042] | [-0.07517880007825879, 0.006526278370887331, 0.058053086784499136] |

B is resampled_ic, A is whole_filter; negative favors resampling. Intervals condition on the fixed TRAIN pool, average the3 optimizer seeds and bootstrap128 held-out mechanisms; they are exploratory, not confirmatory population-level tests. No hyperparameter changed after test opening.

## Crossed-versus-retained interaction

- duffing: 0.0100693; exploratory interval [-0.0587487772531382, 0.07606991640746998].
- stuart_landau: -0.0790363; exploratory interval [-0.23518339204882516, 0.0283588178984873].

The crossed test preserves mechanism/IC marginals but changes state coverage/information, so an interaction alone is not a shortcut proof. High source acceptance and ordinary prior-aware sufficiency limit practical and methodological claims. Selection-aware amortized inference already has close prior art (NOVELTY_ADDENDUM87_V1.md). The scoped new-method route is not supported. This does not falsify all possible effects of trajectory filtering or erase the C86 linear construction.

No new architecture or final research direction is accepted. Any later direction must start from a concrete failure left unresolved by competent ordinary and prior-aware controls.
