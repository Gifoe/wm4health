# Competitive hypotheses and decisions

| ID | Precise hypothesis | Discriminator | Current status |
|---|---|---|---|
| A0 | Identical overlapping raw patches encode identically despite changing window statistics | Original-vs-moving chart, excluding padding | Rejected in the current chart; fixed-chart control exact |
| A1 | Native representation drift proves worse forecasting or lost state information | Held-out native forecast controls with declared interface | Not tested; must not infer from A0 |
| A2 | Cheap transport offers important native end-to-end benefit beyond ordinary causal normalization/re-encoding | Actual speed/error frontier against compiled/recomputed and causal-stat controls | Not established; low priority given direct overlap and small shallow encoders |
| B0 | A shared latent mechanism for alternative futures is absent from prior art | OneWorld plus latent neural-process/Bayesian literature | Rejected as a broad novelty claim |
| B1 | Without paired reset supervision, finite-budget latent belief models can retain calibrated mechanism uncertainty across condition queries under partial observation | Correct ordinary Bayesian/particle and neural-process controls, common data/compute, separately scored within-path and between-query distributions | Unresolved regime-specific question; no new mechanism accepted |
| C0 | Joint action likelihood and controlled dynamics is a new mechanism for two-role actions | DRL Sections 3.2-3.4, proximal causal dynamics literature | Rejected as a broad novelty claim |

For B1, an existing correctly conditioned hierarchical/particle model matching the proposed method's quality and cost falsifies a claimed new advantage. A Gaussian approximation failure alone does not establish a new gap; mixture and particle controls are mandatory. Single-factual training cannot identify arbitrary cross-intervention couplings without declared structural assumptions. Ordinary forecast-only models are not required to output such couplings.

The next useful step is a source-and-protocol comparison of existing amortized mechanism-posterior controls before any new architecture. Training is justified only after specifying an identifiable data regime, an existing method's actual contract and a concrete residual prediction that a strong ordinary control does not trivially resolve.
