# Controlled evidence: separate error sources before proposing a framework

Frozen before these computations. These controls are not a new method, original-paper accuracy reproduction, or proof of a research gap. Parent source qualification is already a separate five-loop interface run.

## A. Gauge-free exact belief reference

Use the native MCAR case1 parameters/fixture from qualification. Compute the exact finite burn-in initial covariance (100 transitions from identity), Q=.1^2 Sigma_d, R=.3^2 Sigma_p. Implement recursive Kalman filtering and, independently, block Gaussian conditioning on the observed coordinates. Compare state means/covariances at t=0/3/7/11 on four recordings. Then transform coordinates by a fixed invertible latent map and verify identical observable future mean/covariance at h=1/4/8. A raw latent-coordinate MSE is not a meaningful cross-model metric. This is privileged known-model truth, not a fair learned baseline and not an assessment of five-loop LIFTS state quality.

## B. Encoder compensation counterexample and ordinary repair

Four-state visible Markov chain, true P=.8I+.2J, misspecified frozen Q=.9I+.1J, uniform J. Compare exact posterior e_s, compensating representation (8/9)e_s+(1/9)pi and ordinary true/refitted transition. Horizons1/4/8. Exact expected Brier excess and KL regret, not finite-sample noise. Purpose: show why a frozen wrong transition can make posterior calibration look harmful, then expose the ordinary refit that removes that motive. The algebraic example is not novelty; PSIM/predictive-state inference already addresses prediction-driven filtering. Do not declare a new framework from a deliberately frozen wrong Q.

## C. Crossed view–dynamics feasibility boundary

Candidate setting only, not an accepted idea: one model reused across observation views and dynamics regimes, test a held-out combination. Train cells (A,d1),(B,d1),(A,d2); shared d1 is state-permutation symmetric, d2 asymmetric. Enumerate all4! observation permutations. Noiseless named symbols and known source transitions are privileged controls; they make the identifiability issue explicit. With zero paired observation anchors, the training observable laws are exactly equal under multiple view alignments while the held-out (B,d2) law differs. A connected cell graph alone is insufficient.

Add fixed paired A/B anchor prefixes of length0/1/2/4/8/16, and condition on held-out B,d2 prefixes1/8/32. All actual observations available to the posterior must be used; do not ignore the new test-prefix information to manufacture failure. Use a full Bayesian mixture over permutations as a strong ordinary control. Over128 prelisted random sequences/permutations, report posterior alignment entropy, exact next-symbol Brier excess and log regret, and exact-law discrepancies. No neural learning, no chosen anchors based on future scores, no test hyperparameters.

This isolates information availability from algorithmic error. If anchors/in-context observations let the ordinary factorized belief model solve the setting, that is a required positive control and closes a naive novelty claim. A future neural framework must demonstrate an additional finite-data/compute interaction beyond this model. No physical-state identifiability claim; only held-out observable-law distinguishability matters. An unseen arbitrary sensor map with no coverage/anchors is not an admissible success target.

V2 numerical correction before any effect row: returned native float32 Sigma matrices have tiny numerical asymmetry. The source generator uses their CPU Cholesky factors; form the effective covariance L L^T in float64 for both exact oracle implementations. Preserve the initial 1.65e-9 mean-check failure. No tolerance, data/model configuration or hypothetical effect was adjusted; the underlying float32 state arithmetic is approximated by the stated linear-Gaussian oracle.
