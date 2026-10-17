# C54: Jointly learned RP-GSSM linear audit

Status: all 12 fixed-budget fits (RP native; ordinary LGSSM EM with the explicitly qualified initial-covariance normalization correction), training-only state maps, causal heldout scoring and independent numerical checks completed. This is a paper-recipe reconstruction using pinned native components; it is not the original full hyperparameter sweep or visual benchmark reproduction.

**Research Question:** Does jointly learned latent recognition recover transition structure that remains useful for causal multi-step prediction, rather than only for full-sequence state readout?

**Evidence:** Complete native-model recipe and strong ordinary LGSSM fits on two in-model synthetic tasks, with frozen training-only maps and oracle conditional means.

| Task | Model | Full smoothing R2, train-only map | Mean eigenvalue error | h=1 state MSE | h=10 state MSE | h=50 state MSE |
|---|---|---:|---:|---:|---:|---:|
|linear_small|rp|0.933015 +/- 0.001018|0.029943|0.188494|0.741257|1.047827|
|linear_small|em|0.960674 +/- 0.000000|0.000769|0.168859|0.697773|0.957503|
|linear_medium|rp|0.744632 +/- 0.007335|0.461281|0.390011|0.867067|1.019076|
|linear_medium|em|0.965215 +/- 0.000000|0.002634|0.170120|0.644126|0.942786|

| Task | Model | h=1 conditional-mean discrepancy | h=10 discrepancy | h=50 discrepancy |
|---|---|---:|---:|---:|
|linear_small|rp|0.0322689|0.0715555|0.073247|
|linear_small|em|3.4704e-05|3.23437e-05|2.00214e-05|
|linear_medium|rp|0.248114|0.199638|0.0972055|
|linear_medium|em|0.000188373|0.000844449|0.000463326|

**Existing Methods:** RP-GSSM and LGSSM EM are established mechanisms. The current comparison cannot support a claim that existing world models generally cannot learn reusable dynamics.

**Proposed Idea:** None accepted. Do not introduce a new architecture from this qualification alone.

**Falsification:** Existing models matching useful conditional predictions refute a broad missing-capability claim. A deficit repaired by ordinary exact-inference LGSSM is model-specific evidence, not proof of a new general gap.

**Feasibility:** Both native implementations ran on the server's existing CPU runtime. Signals, checkpoints and arrays use the authorized large disk. Two synthetic data draws and three model initializations do not establish real-world generalization.

**Next Experiment:** The PRO-DYN source/task audit is complete (PRODYN_SCOPE_AUDIT54.md): it is retained as a forecasting parameterization reference, not a latent-state benchmark. Select the next nonlinear or real-system comparison from an actual released state-learning input contract and a competent existing control. That selection remains open; it is not presented as a selected final research direction. Avoid repeating the fixed-prior alarm diagnostic.

**Decision:** No new direction accepted; the overarching discovery objective remains active.
