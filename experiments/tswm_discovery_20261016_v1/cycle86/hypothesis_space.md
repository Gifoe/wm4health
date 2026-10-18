# Competing explanations

H1 — Whole-realization rejection changes the mechanism population. Directly tested by executing the pinned sampler on paired exact flows. A distribution change alone is an expected consequence of selection, not a research gap.

H2 — Rejection also induces an initial-state/mechanism association that a predictor may use. Snapshot-only probes test association. Mechanism-preserving initial-state resampling tests a learning consequence without changing selected mechanism IDs. Crossed queries additionally keep both mechanism and initial-state marginals exactly fixed. Later trajectory support and information may still change.

H3 — Apparent benefit is merely more data or a larger network. The paired study fixes retained window count, architecture, initialization, minibatch-index streams and update count. It does not fix simulator-attempt costs; changed state distributions are part of the intervention.

H4 — Initial-state resampling universally improves accuracy. Both raw MSE and per-trajectory normalized errors must be inspected. Worsened normalized errors or retained-cohort performance reject an unconditional superiority claim. No test metric or subgroup may be selected afterward to hide this tradeoff.

H5 — A new world-model architecture is necessary. Ordinary local identification is included. Its strong performance limits that claim in this family. Source qualification only requires beating persistence; it does not prove convergence or parity between architectures.

H6 — The selected sampler explains an existing released checkpoint's failure. Not established: exact checkpoint training-data provenance and a competent, controlled retraining comparison are missing. This hypothesis cannot be accepted from source proximity or a small-model diagnostic.
