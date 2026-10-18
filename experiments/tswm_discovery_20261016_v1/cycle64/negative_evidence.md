# Negative evidence and rejected claims

- Reject a new-framework claim from block conditioning bias: causal factorization and feedback identification are established, and the ordinary causal AR is strong here.
- Reject a claim that every frozen model requires retraining: the retained full joint checkpoint supports a valid temporal causal factorization without new fits or data.
- Reject treating the 16 nonintegrable ratios as failure of the exact causal identity. They concern removing one policy from an independently approximated observational density. Compatibility error matters.
- Reject the fixed eigenvalue floor as a solution: in the hardest H64 cell it yields normalized error about 37.3. No floor search or test tuning followed.
- Reject interpreting the SMC threshold failures alone as a gap: H64 iid exact-target references fail that absolute threshold too. Excess moment discrepancy remains in this particular low-budget sampler, with only three MC replications and no full TDS reproduction.
- Reject treating high empirical ESS as a coverage certificate when the population importance-weight second moment is infinite.
- Reject blaming the entire plug-in DDIM error on discretization: exact target scores at identical budgets yield much lower error. Conversely, exact target scores are privileged and are not a practical learned-model solution.
- Reject claiming CaTSG, ADMIT, DiffPO or the longitudinal causal-diffusion paper is empirically defeated: none was executed in C64.
- Reject framing an existing Gaussian information identity, inverse propensity adjustment, g-computation, or posterior SMC as a newly invented mechanism.
- Reject extrapolation to hidden confounding, partial observability, nonlinear learned models, transfer, clinical decision-making or real counterfactual truth. They were not tested.
- Reject counting the 48 horizon fits as 48 independent datasets; they are prefixes of 12 corpora. No GPU/neural training occurred.

Engineering exclusions: H1 has a fixed previous state, so rho/intercept and policy slope/intercept are not separately identifiable. v3 replay uses a fixed QR rank cutoff and checks predictions/noise rather than arbitrary equivalent coefficients. Earlier failed replay attempts changed no scientific setting and created no scored result.
