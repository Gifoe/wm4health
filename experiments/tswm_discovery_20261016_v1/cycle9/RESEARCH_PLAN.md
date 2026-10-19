# Cycle9: native inverse-score falsification

Status: exploratory mechanism test; no accepted research direction. Earlier cycles remain sealed.

Question: does the released action-distance Action-NCE exploit endpoint information about an action sequence when the conditional action mean is zero but a quadratic action functional is known?

Scope: function-level native-loss audit on an explicitly constructed deterministic nonholonomic integrator. This is neither a native visual-control reproduction nor a counterexample to EP-IDM's linear theorem. ACRO's deterministic/exogenous-policy assumptions and finite-horizon coverage must not be silently extended.

Plant: x'=x+a_x, y'=y+a_y, z'=z+(x*a_y-y*a_x)/2. Four-step square loops start at zero, with two orientations and uniformly balanced opposite rotations. Terminal x=y=0 and z=orientation. Actions have unit radius. Endpoint sign is known; individual rotations are not identifiable. Thus per-instance action retrieval has a nonzero irreducible loss.

Frozen before execution: seeds2021/2022/2023; disjoint rotated angle grids for training/validation/test; native temperature0.1; 600updates, Adam0.03; both independent random initialization and exact zero query diagnostics. Native score is extracted as an unchanged AST function from the pinned train.py. Compare an ordinary quadratic-feature contrastive score and analytic class-information ceiling. No benchmark rank or model-performance claim.

Competing explanations: (H1) insufficient optimization; test the convex query objective and zero-gradient/Hessian property. (H2) no information at endpoints; test the known quadratic functional and independent held-out rotations. (H3) all contrastive inverse learning fails; test an existing generic quadratic score. (H4) a novel method is needed; reject if the generic score succeeds. ACRO released action normalization is audited separately without executing its launcher.

Stop rule: a generic score resolving the diagnostic rejects the broad new-method claim. Do not expand this toy into a paper direction without an independent real task, closest-prior novelty audit, and matched deployed-model controls.

Provenance: pinned public Git trees and SHA-verified selected files; only source/configuration downloaded. Computation runs on the existing private RTX5090 server. Raw stdout/stderr, environment, configs, metrics and model states are retained.
