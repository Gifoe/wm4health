# C56 results

4/8 primary a-gradient deviation intervals exclude zero after the prespecified Bonferroni correction. The intervals describe Monte Carlo uncertainty at fixed known parameters, not learned-model generalization.

Primary family:

| r | T | fixed q | B | a-gradient deviation | MC SE | interval low | interval high |
|---:|---:|:---:|---:|---:|---:|---:|---:|
|0.3|6|False|32|-0.11572922|0.0102|-0.14393767|-0.087520761|
|0.3|6|True|32|0|0|0|0|
|0.3|24|False|32|-0.16272176|0.00885|-0.18712588|-0.13831764|
|0.3|24|True|32|0|0|0|0|
|3.0|6|False|32|-0.013847098|0.00176|-0.018704366|-0.0089898297|
|3.0|6|True|32|0|0|0|0|
|3.0|24|False|32|-0.032112223|0.00232|-0.038512041|-0.025712405|
|3.0|24|True|32|0|0|0|0|

Descriptive native full-gradient reference-size sweep (95% intervals outside the B=32 primary family):

| r | T | fixed q | B | a-gradient deviation | MC SE | interval low | interval high |
|---:|---:|:---:|---:|---:|---:|---:|---:|
|0.3|6|False|4|-0.59769489|0.036|-0.66855959|-0.52683019|
|0.3|6|False|16|-0.20375768|0.0146|-0.23259411|-0.17492126|
|0.3|6|False|32|-0.11572922|0.0102|-0.14393767|-0.087520761|
|0.3|6|False|128|-0.044285381|0.00507|-0.054269787|-0.034300974|
|0.3|6|False|512|-0.013173175|0.00256|-0.018217589|-0.0081287606|
|0.3|24|False|4|-0.70943135|0.0243|-0.75727936|-0.66158334|
|0.3|24|False|16|-0.27736257|0.0114|-0.29982044|-0.25490471|
|0.3|24|False|32|-0.16272176|0.00885|-0.18712588|-0.13831764|
|0.3|24|False|128|-0.051262555|0.00444|-0.060014054|-0.042511055|
|0.3|24|False|512|-0.015573879|0.00258|-0.020661337|-0.010486422|
|3.0|6|False|4|-0.10292819|0.00766|-0.11801417|-0.087842199|
|3.0|6|False|16|-0.028278761|0.0028|-0.033783197|-0.022774326|
|3.0|6|False|32|-0.013847098|0.00176|-0.018704366|-0.0089898297|
|3.0|6|False|128|-0.0041181505|0.000799|-0.0056922115|-0.0025440894|
|3.0|6|False|512|-0.00081216172|0.000376|-0.0015516512|-7.2672253e-05|
|3.0|24|False|4|-0.19669559|0.00913|-0.21468477|-0.17870641|
|3.0|24|False|16|-0.057003292|0.00341|-0.063709144|-0.05029744|
|3.0|24|False|32|-0.032112223|0.00232|-0.038512041|-0.025712405|
|3.0|24|False|128|-0.0088892891|0.00107|-0.010992059|-0.0067865187|
|3.0|24|False|512|-0.0020899624|0.00059|-0.0032526015|-0.00092732329|

There are 144 diagnostic rows (24 data-law/reference-size/jitter cells, two gradient conventions and three reference laws), 432 parameter summaries and 256 independent Monte Carlo banks per cell. Reference prefixes and target sequences are paired across controls. These are not neural training runs or real datasets. Full numeric records include population gradients under literal jitter, raw and integrated-target gradients, recognition blocks and all uncertainty estimates. No optimizer updates were performed.

This is a scalar realizable Gaussian objective test. It does not establish the cause or effect size of C54 multidimensional learning failure, nonlinear applicability, a practical repair, or originality. The existing EM option changes recognition as well as prior learning; cancellation of its direct a-gradient deviation is not evidence that its full training succeeds. No final research direction is accepted.
