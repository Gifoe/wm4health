# Competing hypotheses — working cycle16

| Hypothesis | Discriminator | Current standing |
|---|---|---|
| H1: the native cross-scale moment approximation has substantial learned forecast consequences | Final selected-checkpoint joint vs dense-independent covariance diagnostic, unchanged mean/decoder | Pending; source-module witness alone insufficient |
| H2: spatial covariance projection explains most of any difference | Dense-independent control vs native projection | Implemented; final evaluation pending |
| H3: learned dynamics/noise/decoder absorb the approximation | Learned covariance contribution and decoded sensitivity after the fixed fit | Pending; early smoke does not show an extreme output effect |
| H4: apparent failure is inadequate training or poor transfer | Fixed validation curve, ordinary references, raw logs; no diagnosis from budget exhaustion alone | Training active |
| H5: ordinary joint/particle inference is already adequate | Cycle15 measured dense inference cost, then trained relevance/quality controls if warranted | Strong negative evidence against a specialized compression architecture; no general solution claim |
| H6: a task-latent posterior itself is novel | Primary RAMP, NP/MPNP, manifold system-ID and Bayesian-dynamics review | Rejected as a broad novelty claim |

No hyperparameter is selected from the held-out test segment. Initial fast/context covariance is set to zero in both dense controls; true learned joint posterior is unknown. A post-hoc decoder effect cannot discriminate all hypotheses. Any consequential residual needs an independently frozen confirmation protocol and appropriate trained baselines before new method design.
