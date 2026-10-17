# C56 true-point finite-reference audit

4/8 primary a-gradient deviation intervals exclude zero after the prespecified Bonferroni correction. The intervals describe Monte Carlo uncertainty at fixed known parameters, not learned-model generalization.

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

The primary intervals are two-sided family-wise 0.05 intervals for eight a tests. Full gradients use the paired exact-reference control variate; raw gradients, recognition blocks, all reference sizes, external and leave-one-out controls remain in the JSON and CSV. Fixed-q a deviations cancel structurally (maximum absolute mean 0); they do not validate the full EM learning trajectory.

This is a scalar realizable Gaussian objective test. It does not establish the cause or effect size of C54 multidimensional learning failure, nonlinear applicability, a practical repair, or originality. The existing EM option changes recognition as well as prior learning; cancellation of its direct a-gradient deviation is not evidence that its full training succeeds. No final research direction is accepted.

See PROTOCOL.md, DERIVATION.md and outputs/QUALIFICATION56_V1.json. Run qualify56_v1.py, run56_v1.py and analyze56_v1.py in a fresh output directory with the pinned source/runtime. start56_v1.ps1 durably runs qualification and Monte Carlo stages. Source pins, exact bank hashes and all raw replicate arrays are retained. Large arrays stay on configured server storage; no local result files were synchronized.
