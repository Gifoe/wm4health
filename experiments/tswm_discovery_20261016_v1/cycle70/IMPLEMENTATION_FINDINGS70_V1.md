# Reproducible implementation findings

Pinned upstream: https://github.com/gatsby-sahani/rp-gssm/tree/a75225d5edceb972bb872dc9a335608733567dd4

1. `ACLGParam.init` in rp_ssm/dists.py:422 calls `action_mapper.init(key, actions)`, while `DiscreteChoiceActionMapper.init` in actions.py:56 requires dimension and transition-mode arguments too. Direct initialization raises the retained TypeError. `DiscreteSignatureBridge` forwards those explicit arguments to the unchanged native mapper. This is interface binding, not a learned replacement.
2. `ACLGParam.update` at dists.py:434 does not carry the original `start_from_invariant` flag (or explicit opt_params). All12 noninvariant cells reset their returned initial mean/covariance to the default; reported errors retain the original output. `preserving_update` supplies the original flags to the same constructor. The12 default-invariant cells retain their flag. Only these exercised settings are qualified; no blanket claim that every optimizer option is fixed.
3. With the continuous mapper, `transition_matrix='constant'` / `transition_bias='none'` still lets the returned C/c affect the chain. A fixed zero-versus-one action control changes mean by 0.0175799157111 and covariance by 5.99853976664e-05. A separate mapper output mask enforcing those disabled terms produces exactly equal arrays. The full original and masked arrays remain in the hashed control archive.

No upstream file, old experiment or shared environment is modified. These repairs do not establish failure of representation learning, need for a nonlinear prior, or a new world-model architecture.

