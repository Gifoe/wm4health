# Statistical boundary and technical repair

[SNVI, section3.4 and appendixA.6](https://arxiv.org/html/2203.04176v3) already corrects likelihood-estimation bias from excluding invalid simulations using an estimated validity factor. [The SBI practical guide, appendixA1](https://arxiv.org/html/2508.12939v1) distinguishes how posterior, likelihood and ratio estimation handle invalid outputs. A validity classifier or inverse acceptance weighting is therefore not a new contribution.

A narrower question worth testing is selection followed by observation coarsening. Let H denote the available history and S the complete-trajectory retention event. The elementary conditioning identity is

    p(theta | H,S=1) proportional to p(theta | H) * P(S=1 | theta,H).

If retention is fully determined by H, that factor cancels across theta. If retention also depends on an unseen future or hidden/masked coordinates, it need not cancel. This is standard selection/missing-data reasoning, not a new theorem and not a contradiction of SNVI's stated result. For independent initial conditions, a whole-bag rule also changes the mechanism marginal by a factor alpha(theta)^P, with alpha the single-path acceptance probability. C86 measures that effect rather than presenting it as novel theory.

For a potential world-model contribution, the missing evidence is whether a real pretraining pipeline performs this non-ignorable selection/coarsening, whether its competent model learns a harmful association, and whether a practical correction improves robust simulation beyond ordinary baselines. In C86, posterior-style interpretations are explanatory only: the trained predictors minimize future MSE, not a posterior-density objective. The current results do not prove posterior miscalibration.

The sampler verifier's initial direct HTTPS request failed because the server's outbound route refused the connection. Public source bytes were instead fetched with normal verified HTTPS into local RAM, compared byte-for-byte to the pinned server cache through pinned SSH, and a receipt was written only on the server. Verifier v2 consumes that receipt and retains all scientific assertions/tolerances from v1. No payload files were synchronized locally, and no network/global-package setting was changed.
