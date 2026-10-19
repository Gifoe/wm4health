# Competing hypotheses after the native gate

| Hypothesis | Discriminator | Status |
|---|---|---|
| Released model is a weak or context-blind forecaster | Frozen native accuracy and wrong-system context | Contradicted on this native system family |
| A single variance temperature explains trajectory coverage | Separate-system temperature, fixed probe | Contradicted for total functional |
| Ordinary cross-horizon residual covariance explains much of the apparent joint failure | Same means/marginal variances, diagonal versus full | Supported; NLL improves-.15058 and total coverage51.95→92.19% |
| A new dynamics-specific method is required | Strong functional/generative baselines and prospective generator | Not established |
| Remaining conditional gap comes from native future normalization | Rebuild a separate prospective generator, freeze train-visible scales, validate its changed task | Untested; do not compare it as an identical native protocol |
| Efficient belief updates / joint sampling are themselves missing methods | Primary incTNP, AR-CNP and causal-buffer papers | Generic version already covered |

No new architecture is selected. Before another method experiment, require a residual failure that existing Gaussian/functional/generative controls cannot explain and that is consequential under a valid deployment contract. Existing point-only methods need not expose Bayesian interfaces.
