# Continuous partially observed counterfactual computation: falsification pilot

This is a new controlled computational experiment, not a reproduction, not an
accepted research idea, and not a claim to full nonparametric causal bounds.

We restrict the admissible SCM family to a known two-action linear Gaussian state
model with one shared, time-invariant orthogonal innovation alignment R in O(d).
All family members have identical observational and interventional transition
kernels. No actions depend on unobserved noise. Initial-state law, transition and
observation matrices, process covariances and sensor-noise covariances are known.
The factual observations are noisy and partially observed; the counterfactual
action plan is fixed before observing the episode. The query is conditional
terminal squared error, not optimal adaptive control or individual causal truth.

Competing explanations:

1. Continuous partial observability leaves a costly horizon-dependent
   counterfactual optimisation after abduction.
2. Ordinary Gaussian sufficient-statistic compression and an existing generic
   orthogonally constrained quadratic SDP already resolve the small regime.

Haugh and Singal (ICML2023) already reuse one factual posterior sample set across
causal models. That reuse is NOT new. Gaussian conditioning, Kalman filtering,
orthogonal quadratic optimisation, Shor relaxation and moment matrices are also
established. We test whether any computational residual survives these controls.

Frozen cases: d in {2,3,5}, horizon in {64,256}, data seeds {61,62,63}, three
independent generated systems, giving 18 cases. Do not tune on these outcomes.
For each: exact query-feature posterior, direct twin-state Kalman cross-check for
20 fixed random orthogonal R (both determinant components), generic SDP lower
and upper bounds, feasible multistart optimisation and random rounding. For d=2,
both O(2) components admit a trigonometric degree-two global cross-check.

Record numerical primal/dual residuals; numerical SDP answers are not formal
interval-arithmetic certificates. If the generic SDP's feasible-to-relaxation
gap is <=1% of max(1, absolute extremum) in all cases, stop the custom-solver
direction in this regime. A gap alone does not establish research novelty.
Any candidate continuation must also show a meaningful native-model task,
stronger generic verification baselines and a closest-prior novelty boundary.

No model fitting, no test-set hyperparameter search, no publication authorised.
