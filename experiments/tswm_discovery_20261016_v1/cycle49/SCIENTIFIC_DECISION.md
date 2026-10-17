# C49 scientific decision

Decision: DEPRIORITIZE_CURRENT_RECIPE

Accepted final directions: 0

The current observation-marginalized conditional-score recipe does not provide a stable general quality/cost advantage over complete-clock variational learning of the same predictive core. This completed screen supports lowering its priority. It does not rule out conditional stochastic-world modeling or observation-aware learning generally.

## Verified primary evidence

Eight fresh fits /18800 updates.16 raw evaluation groups /48 summary groups /12288 model-context-reveal scores;512 independent synthetic physical paths,1536 paired sensor-density histories,128 joint future samples per prediction. No independent real domain was evaluated.

Equal-weight means across the three pre-specified sensor densities and two fitted seeds, open-loop24-bin physical energy (lower is better):

| Process | Observation-averaged score | Complete-clock ELBO | Score minus ELBO |
|---|---:|---:|---:|
| Rotating | 0.2984525914 | 0.2936971191 | +0.0047554724 (+1.62%) |
| Bistable | 0.3071688372 | 0.3024564767 | +0.0047123605 (+1.56%) |

Rotating differences are adverse in both fitted seeds: +0.00546158 and +0.00404937. Bistable changes sign: +0.01330112 and -0.00387640. Updated12-bin means also do not favor the score recipe, but before/after update benefits must use identical final12 targets; full24 versus12 is not a valid update comparison.

All original future predictions, noise draws and energy/MSE/observed scores were independently replayed. Max model discrepancy1.59899e-4 under pre-frozen atol/rtol3e-4; energy arithmetic4.44e-16; observed-score discrepancy1.82e-7. Native predictive initialization and training-index schedules match within case/seed. Verification optimizations changed I/O, equivalent numerical evaluation and independent-group scheduling only; primary fits/scores were not rerun.

## Pre-registered fixed-time evidence

The60/120/180-second rule was sealed before primary EVAL opening. All72 summary groups /18432 model-context-budget entries were reported;22 unique weight evaluations, reusing identical verified primary results when appropriate. New frontier predictions underwent independent model/energy/MSE replay.

| Process | 60s mean energy delta | 120s mean energy delta | 180s mean energy delta |
|---|---:|---:|---:|
| Rotating | +0.00668030 | +0.00363422 | +0.00475547 |
| Bistable | +0.00907795 | -0.00169634 | +0.00471236 |

Negative deltas favor the score recipe. Bistable120s is a small favorable mean; its episode-bootstrap interval[-0.00795610,+0.00463150] crosses zero. These intervals keep density views together and condition on the two fitted model seeds; they are not uncertainty across training seeds or real domains. No single budget is selected from EVAL.

## Preserved exploratory rescue, not a selected method

Three of18 two-seed-mean process/budget/density cells favor the score recipe, all at density0.25: rotating60/120s and bistable120s. Some sparse-budget differences agree in both seeds, but the benefit does not persist uniformly with training budget or process. Preserve this as a hypothesis about sparse-observation calibration/selection, not as a demonstrated new mechanism or main contribution.

If an independent real-domain failure later makes this hint important, first distinguish ordinary observation-coverage augmentation, validation-policy choice and approximate-inference constraints from analytic averaging. A calibrated thinning-augmentation or competent observation-aware filter is a necessary ordinary control. Do not tune loss weights or select the120s/density0.25 subgroup on this EVAL.

## Novelty and attribution

The score is a pullback/integration of an existing kernel score through a specified observation kernel. Conditional Neural SDE scoring, Ambient methods,2026 convMMD/deconvolution generative learning and observation-aware SBI already cover major ingredients. A useful distinct learning interaction could still contribute, but this screen does not demonstrate one. The two-arm comparison also changes objective geometry and training-only future recognition, so it cannot isolate analytic averaging as the causal explanation of a difference.

One maximum-budget fit and three final-step selections limit convergence claims. Two controlled2D process families with a known calibrated sensor do not establish broad real-world utility or physical latent identification.

## Next action

Return to source-grounded real-data failure discovery before adding another loss. The pinned native TSWM Greenhouse preprocessing has now failed an input-boundary contract on a constructed window accepted by its own missingness filter: changing only future target values changes the imputed observed history. Measure real-data prevalence before interpreting a baseline comparison; preserve original-protocol reproduction separately from any information-safe correction. This is an implementation issue, not a new research direction.

No final research direction is accepted. Overall discovery goal remains ACTIVE_UNACHIEVED.
