# Numerical qualification, not learned-model performance

The predeclared factorial grid is3 seeds × continuous/discrete actions × additive/multiplicative transitions × invariant/noninvariant initial state =24 cells. Each cell has4 sequences of32 two-dimensional states. Likelihood precision is either.02 or1. The32-step full smoother is a component check; only the first16 observations are supplied for the separate future-plan test. Both future plans share the same inferred state and frozen parameters. No future observation enters that inference.

The original author's smoother explicitly adds1e-5 I to marginal covariances before computing adjacent cross-covariances. Raw outputs are preserved; report marginal comparison after removing that declared pad and raw cross-covariance discrepancy separately. Thresholds were1e-7 for mean/marginal/moment/forecast checks and1e-4 for raw cross-covariance. This permits no claim of exact equality of the returned padded joint posterior.

| Cell | posterior mean error | covariance error after explicit padding | raw cross-covariance error | future mean error |
|---|---:|---:|---:|---:|
|continuous_additive_invariant1_seed70001|1.6236272e-09|1.2263001e-09|1.2908618e-05|5.5511151e-17|
|continuous_additive_invariant0_seed70001|2.2138418e-09|1.186918e-09|1.2727998e-05|2.7755576e-17|
|continuous_multiplicative_invariant1_seed70001|1.3128271e-09|1.4629688e-09|6.4995923e-09|0|
|continuous_multiplicative_invariant0_seed70001|1.3128267e-09|1.4109506e-09|8.7316621e-09|0|
|discrete_additive_invariant1_seed70001|2.8288247e-09|1.284617e-09|1.3601027e-05|5.5511151e-17|
|discrete_additive_invariant0_seed70001|2.9377375e-09|1.2419274e-09|1.3376697e-05|2.220446e-16|
|discrete_multiplicative_invariant1_seed70001|2.5536946e-09|1.2649628e-09|1.334551e-05|1.110223e-16|
|discrete_multiplicative_invariant0_seed70001|2.6793338e-09|1.2231806e-09|1.3135487e-05|1.110223e-16|
|continuous_additive_invariant1_seed70002|2.6921463e-09|1.2295881e-09|1.2908362e-05|5.5511151e-17|
|continuous_additive_invariant0_seed70002|2.415345e-09|1.1899495e-09|1.2727771e-05|2.7755576e-17|
|continuous_multiplicative_invariant1_seed70002|1.5416465e-09|1.4629688e-09|3.2780527e-09|0|
|continuous_multiplicative_invariant0_seed70002|1.5416465e-09|1.4109506e-09|3.0538157e-09|0|
|discrete_additive_invariant1_seed70002|2.5543173e-09|1.2871891e-09|1.3601059e-05|5.5511151e-17|
|discrete_additive_invariant0_seed70002|1.996928e-09|1.2443551e-09|1.3376727e-05|1.110223e-16|
|discrete_multiplicative_invariant1_seed70002|2.5726268e-09|1.2678533e-09|1.3345544e-05|8.3266727e-17|
|discrete_multiplicative_invariant0_seed70002|2.09623e-09|1.2258919e-09|1.3135518e-05|5.5511151e-17|
|continuous_additive_invariant1_seed70003|2.8225522e-09|1.2077417e-09|1.2908564e-05|5.5511151e-17|
|continuous_additive_invariant0_seed70003|1.6155419e-09|1.1692169e-09|1.2727972e-05|5.5511151e-17|
|continuous_multiplicative_invariant1_seed70003|1.889755e-09|1.4629688e-09|9.1052097e-09|0|
|continuous_multiplicative_invariant0_seed70003|1.2237775e-09|1.4109505e-09|1.2447584e-08|0|
|discrete_additive_invariant1_seed70003|1.8482735e-09|1.2640216e-09|1.3601034e-05|1.110223e-16|
|discrete_additive_invariant0_seed70003|1.8651185e-09|1.2223102e-09|1.3376704e-05|1.110223e-16|
|discrete_multiplicative_invariant1_seed70003|2.0383899e-09|1.244606e-09|1.3345518e-05|1.110223e-16|
|discrete_multiplicative_invariant0_seed70003|2.0484006e-09|1.2037985e-09|1.3135494e-05|1.110223e-16|

Independent replay used96 full smoothers and96 prefix smoothers, verified15 source blobs and25 numeric archives, and freshly repeated one native cell. Kalman/RTS versus dense posterior maximum mean error is 3.10862446895e-15; covariance 7.77156117238e-16. Native mean differs from RTS by at most 2.93773783078e-09; raw adjacent cross-covariance error reaches 1.36010591589e-05. These are numerical checks on known parameters, not96 independent learning experiments. Qualification runtime 62.100s; replay 12.192s, CPU only.

