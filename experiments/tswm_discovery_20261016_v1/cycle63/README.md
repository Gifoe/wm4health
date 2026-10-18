# C63: COFE native probability/objective qualification

Three untrained initialization fixtures, four early training-region contexts each, original L96/H720 source dimensions. CPU float64. Zero trained models, zero optimizer steps, zero heldout scores. This is component qualification, not paper-performance reproduction. Raw inputs, weights and traces stay on G.

The auxiliary noise estimator can be set identically to zero without changing any reconstruction, predicted location/scale, or variational distribution in the checked same-RNG calls. Its penalty then attains its analytic global lower bound, and the auxiliary-driven gradients into both the estimator and encoder vanish. Thus this implemented penalty alone does not rule out a collapsed estimator. This is not proof that trained COFE converges to collapse or that its theoretical identifiability result is false.

At N=sample_schedule=20 the native method returns component locations and omits positive within-component observation variance. At N40 the same latent-component parameters produce Normal samples. Official ETTh1 N100 does not enter the special branch; this audit does not invalidate its published scores. Numerical differences in finite sample quantiles must be distinguished from this explicit branch changing the component law.

No new world-model research direction is established. Zero-head collapse prevention by an invertible normalized innovation density/Jacobian is established probability and flow machinery, not a new method. Do not claim novelty by adding it to this code. Before treating this source as a mechanism baseline, separate original-protocol use from any scientifically justified correction.
