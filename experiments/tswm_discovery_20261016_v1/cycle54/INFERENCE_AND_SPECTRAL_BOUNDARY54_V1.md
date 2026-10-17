# Inference and spectral-learning novelty boundaries

This is a targeted primary-source/code review and an analytic input-semantics fixture. It is not a new trained method or a formal SelfTICA reproduction. The ongoing C54 comparison remains the immediate empirical decision.

## Existing results that constrain generic proposals

| Primary source | Verified point | What it prevents us claiming | Scope limit |
|---|---|---|---|
| [Turner and Sahani, author-hosted chapter draft](https://www.gatsby.ucl.ac.uk/~maneesh/papers/turner-sahani-2010-ildn.pdf); [Cambridge2011 chapter record](https://www.cambridge.org/core/books/abs/bayesian-time-series-models/two-problems-with-variational-expectation-maximisation-for-time-series-models/31D8D42F70970751C86FEC41E0E7E8D3) | Variational restrictions can bias learned time-series parameters; a tighter posterior approximation need not yield better parameter learning. | Merely observing approximation-induced dynamics bias is not a new research problem. | Its variational examples are not a proof that C54's particular RPM reference-normalizer mechanism is identical. Native Gaussian inference is exact for the implemented surrogate, not necessarily the proper RPM posterior. |
| [Frazier et al., arXiv2106.12262v3](https://arxiv.org/html/2106.12262v3), sections3 and5.3 | State/parameter inference and predictive accuracy can disagree; approximation of local states can affect global consistency and predictive scores. | A generic claim that prediction accuracy does not ensure correct latent inference is already established. | Their expanding-window predictive exercise evaluates one-step densities over different numbers of out-of-sample cases. A longer evaluation period must not be misreported as a longer multi-step forecast horizon. |
| [SelfTICA, arXiv2606.15495v3](https://arxiv.org/html/2606.15495v3), revised9Sep2026, sectionsIV.2/IV.4/IV.5 | Joint encoder/predictor spectral contrastive learning is followed by train-representation TICA; the paper distinguishes equilibrium relaxation times from projections learned using biased exploratory dynamics. | Adding a spectral head, a post-training timescale analysis or reusable contrastive features is not by itself new. | These molecular representation/sampling results are not a blanket solution to arbitrary noisy, partially observed probabilistic world modeling. They were reviewed, not reproduced here. |

## Actual SelfTICA implementation reviewed

The companion repository is pinned at [855c337830b16b8cc83e050ef5ad55947baedc95](https://github.com/Kai-Zhu-2001/SelfTICA/tree/855c337830b16b8cc83e050ef5ad55947baedc95), committed4Oct2026. Its reproduction guide points to frozen paper training code in the associated data archive. We did not download that archive, launch molecular simulators or claim that the latest branch reproduces the frozen paper results.

The current mlcolvar release/2.0 branch is separately pinned at [3a2549ffa63e47b30c390eebc84bed588c7d3868](https://github.com/luigibonati/mlcolvar/tree/3a2549ffa63e47b30c390eebc84bed588c7d3868), committed13Oct2026. The actual [SelfTICA model](https://github.com/luigibonati/mlcolvar/blob/3a2549ffa63e47b30c390eebc84bed588c7d3868/mlcolvar/cvs/timelagged/selftica.py) instantiates the L2 spectral mode of [ContrastiveLoss](https://github.com/luigibonati/mlcolvar/blob/3a2549ffa63e47b30c390eebc84bed588c7d3868/mlcolvar/core/loss/contrastive_loss.py). Its selected objective uses positive pair alignment and off-diagonal squared similarities, rather than the InfoNCE exponential normalizer. TICA is computed separately. This is an existing alternative family to review if a future comparison concerns representation learning under a reference-based loss; it is not the same probabilistic observation model as RP-GSSM.

The source and paper also motivate a necessary interface check: a contrastive/spectral predictor need not equal a conditional-mean transition. Rolling it out without establishing its meaning can manufacture an apparent dynamics failure. The following simple fixture makes that distinction exact.

## Analytic fixed-feature fixture: critic versus transition

Take a stationary two-dimensional linear Gaussian process with A=diag(.9,.6), state covariance C=diag(4,.25), and process covariance Q=C-A*C*A^T=diag(.76,.16). Q is positive, and the physical conditional-mean transition is A.

For fixed, centered features and the unregularized population L2 bilinear contrastive objective, with score (P*z)^T*z_next, the independent-negative term is tr(P*C*P^T*C), while the matched-pair term is -2*tr(P*C*A^T). The unique optimal fixed-feature critic is P=C^(-1)*A. Here it has eigenvalues(.225,2.4), despite the true transition being stable. The ordinary covariance conversion C*P recovers A exactly. This follows from the spectral objective; it is not a new algorithm or a measured failure of a trained SelfTICA model. The regularized, finite-batch, jointly learned model must retain its own contract.

Run scripts/check_spectral_critic54_v1.py for deterministic arithmetic, gradient and Hessian checks. The fixture uses no training data, optimizer update, neural fit or C54 heldout artifact. It supports one practical exclusion: do not use an unconverted spectral critic as a physical rollout baseline and then treat its error as a new world-model deficiency.

## Consequence for the active research search

The broad slogans above are excluded as stand-alone innovations; potentially useful combinations are not rejected merely because their components are known. A framework-level contribution still needs a consequential capability failure that survives correct input semantics, competent existing source settings, ordinary likelihood controls and relevant compute/information comparisons. It then needs an actual useful mechanism validated on fresh heldout evidence beyond two in-model linear tasks.

The present C54 source-option fits, ordinary corrected EM and exact causal oracle will decide whether to continue the current lead. The already executed DCL core qualification is not a scientific benefit; SelfTICA is an additional source-reviewed control family, with0 executed fits here. No new architecture, benchmark superiority or accepted final direction is claimed by this note.
