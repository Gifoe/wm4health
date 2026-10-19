# Prospective numerical gates, before generating cycle24 data

Use original float32 native decoder/encoder, but float64 quadrature, likelihood normalization and metric reductions. No test-based scientific choices. Native latent Gaussian quadrature is an approximation; report128-vs64 physical moments and dense-grid reference rather than assuming exactness.

The7-sigma latent interval omits tiny prior probability, but tiny probability can matter to a low likelihood held-out outcome. Therefore, in addition to the already planned8193-vs16385 grid check, compare[-7,7] to[-10,10] with16385 points on the fixed first8 contexts for actual predictive mixture log density. Report a failed truncation check and repair numerical integration in a distinct attempt before interpreting scores. This additional integration check is specified before data generation, not selected from observed scores.

Diagnostic thresholds for interpretation: maximum actual-query grid-refinement error .01 nats; latent tail sensitivity .01 nats; query first-moment integration error .01 prior standard deviations; native broadcast-vs-original-tensor5e-6 absolute. Numerical excess is not a world-model research failure. No-update always passes a tower test and remains an explicitly uninformative control. The physical posterior is privileged, evaluated with the complete observed context prefix; it is not a fair learned-model architecture competitor.

This is the same fixed benchmark/model/data protocol if a computational quadrature resolution needs repair. No new loss or checkpoint can be selected to improve test scores.
