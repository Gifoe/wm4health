# Competing hypotheses

| Hypothesis | Falsification/control | Judgment |
|---|---|---|
| H1: irregular time is the dominant physical-simulation gap | Same dynamics information comparison, measured interval prediction, actual-time conditional arm | Not supported as dominant; improvements are conditional/mixed and literature is established |
| H2: an inferred latent state controls when RPM is read | Firmware interrupt/getter audit | Proposed blocking-read mechanism rejected; other clock mechanisms not identified |
| H3: unmodeled nonlinear plant laws explain all OOD residual | Existing flexible HGB plus per-regime diagnostics | Not established; residual is concentrated in changed sensor resolution |
| H4: acquisition quantization explains much observed score growth | Inverse-RPM period grids, error contribution, float32 source arithmetic | Strongly supported as competing measurement explanation; exact hardware causation remains unproven |
| H5: a modern predictor requires a new architecture | Existing ridge/persistence/HGB, timing and measurement controls | Not shown; no new method should be built on this evidence |
| H6: inverse-action sufficiency extends to nonlinear reachability | New primary literature/assumption audit and then matched native test | Untested fresh question; not a conclusion of wind-tunnel experiments |

No new hypothesis is accepted just because a prior paper lacks a universal nonlinear theorem. For H6, distinguish ordinary first-order controllability, endpoint action reconstruction and nonlinear reachable directions. Read native definitions and compare known reconstruction/control objectives before running another toy. A classical counterexample or a loss-minimizer construction alone is not evidence of a practical unresolved research problem.
