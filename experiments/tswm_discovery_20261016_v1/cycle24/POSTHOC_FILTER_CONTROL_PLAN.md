# Post-hoc ordinary-inference controls

Written after the completed first48-cell outcomes were inspected and the quadrature repair finished; before running these controls. This is diagnosis on the same inspected cohort, not independent confirmation or a proposed new method.

Motivation: ordinary dense one-dimensional likelihood conditioning removes most internal drift, but does not uniformly improve real query scores over the learned updater. A new framework is not justified if a cheap standard approximation already provides the useful tradeoff. Recent variational manifold system identification (arXiv2607.18965) explicitly combines amortized initialization and MAP/Laplace refinement; generic amortized-to-iterative inference is therefore already prior art.

Keep all six checkpoints, data,48 conditions, prior q(beta|C), likelihood, and query unchanged. Add two established controls:

1. EKF: native decoder Jacobian at prior mean; linear-Gaussian one-observation update using both output coordinates and the original observation variance. No trained parameters.
2. MAP + generalized Gauss-Newton Laplace: initialize at the native post-evidence encoder mean, optimize the same prior-times-likelihood with32 damped scalar Gauss-Newton steps, trust radius3prior standard deviations and8 backtracking halvings. Covariance is inverse positive Gauss-Newton posterior precision. This is a standard local Gaussian approximation, not exact Bayes. Compare attained objective with the existing8193-grid mode and record optimization gaps; do not silently call a poor local optimum a competent baseline.

Measure full physical predictive mixture scores on the original actual queries, prior/probe update time excluding offline mixture integration, and grid/tail checks. No hyperparameter search, new loss, retraining, native checkpoint modifications or claims of innovation. These results can weaken the case for new architecture; a positive post-hoc result requires fresh confirmation before a research claim.
