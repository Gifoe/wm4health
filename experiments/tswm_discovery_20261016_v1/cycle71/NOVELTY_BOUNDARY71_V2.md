# Reuse candidate: new overlap checks, 18 October 2026

This is a primary-source literature addendum, not a new experimental result or accepted research direction. It refines candidate A and challenges candidates B/C in research_ideas.md.

[Platonic Representation Hypothesis on World Models, v2](https://arxiv.org/html/2608.23720v2), sections 4.4 and B.2, already studies functional reuse between separately trained DINO-WM predictors. It freezes the original components and trains lightweight MLP adapters, evaluating the stitched predictors through planning. Thus, independent pretraining followed by lightweight latent stitching is existing work. Its visual planning setting does not prove that arbitrary numerical time-series modules are interchangeable. That difference alone would only motivate a transfer study, not establish a new method.

[BRICKS-WM](https://arxiv.org/html/2606.16489), sections 4.2/4.4 and 6, already learns a latent interface for reusing frozen background dynamics and includes a residual adapter. It assumes a directional agent-to-background generative factorization, inherits perception during transfer, and explicitly acknowledges richer reciprocal interactions as a limitation. Its own sufficient condition concerns the conditional interface process, not merely matching a Gaussian marginal. A generic interface regularizer, frozen module, or adapter cannot be presented as new here. We have not executed BRICKS-WM or located an author implementation in the inspected primary pages; neither fact constitutes a scientific gap.

These observations are literature-based. They are not evidence that a trained model fails on this project's data. No broad claim about all world models follows.

## Remaining question and competing explanations

A narrower development question is whether learned modules can retain useful predictive meaning when recomposed across both observation interfaces and reciprocally coupled mechanisms. This remains unspecified at the method level. It must survive all of the following alternatives:

1. Ordinary observation calibration plus a competent relational world model suffices. C41-C46 already support this possibility in their controlled setting.
2. A recurrent state observer or direct joint model supplies the missing state information; the deficit is observability or optimization, not modularity.
3. Ordinary adapter training or a graph message-passing simulator handles compatibility. The papers above and Modular NRI make these required controls.
4. The apparent failure results from finite-step simulation error, an inadequate graph search, capacity, or insufficient training. These must be separated before proposing a new mechanism.

Adding a reverse edge, a graph network, a continuous-time solver, or an alignment loss is not itself a contribution. Likewise, the absence of a combined experiment in one paper does not establish a research gap. No restriction such as zero-shot transfer or no calibration is imposed merely to manufacture failure.

## Next decisive evidence

Finish the already running three-seed source-based modular baseline. Preserve its native reset-block validation metric for the declared checkpoint selection. Qualify a separate common-history free simulator, then compare complete fitted models and ordinary history-based system identification on development data under a frozen protocol. Reserve independent structural compositions for an explicitly separate experiment: row-disjoint source trajectories do not certify unseen mechanism combinations.

Do not promote either independently trained module stitching or bidirectional module communication as a new idea based only on this search. No final direction is accepted.
