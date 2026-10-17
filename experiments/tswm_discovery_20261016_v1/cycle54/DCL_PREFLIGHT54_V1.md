# Native DCL core qualification, not a formal reproduction

Purpose: establish a competent second decoder-free dynamics-learning executable before deciding whether any RP-specific phenomenon merits a broader comparison. No trained-method superiority, real-domain evidence or accepted new research direction is claimed.

Pin dynamical-inference/dcl at2cd46d13ae9e27feeefafe5745f26003929f9c6e. Verify the53 selected core/license/configuration files against their Git blob hashes in dcl_source_manifest54_v1.json. The service-specific DataJoint implementations, .env files, sweeps and notebooks are excluded. Native core files are unchanged. Do not connect a database or invoke external experiment tracking.

Create a separate project venv with shared existing Torch2.8.0+cu128; install native pinned typeguard2.13.3 and source-allowed jaxtyping0.2.38 there only. Record actual package versions and source hashes. Do not mutate the current RP/JAX runtime, system libraries or drivers. Cache downloads on the authorized large disk.

Qualification uses deterministic small synthetic observations in memory, not the released paper datasets or C54 heldout data. Run native full-denominator MSE InfoNCE in float64 and compare loss and reference gradients with independent NumPy equations, then compare native linear dynamics with NumPy matrix multiplication. Check native MSE-based switching logits against their exact documented reciprocal-MSE equation; changing only the supplied future target may change retrospective mode preference. This is an input-contract demonstration, not a causal forecasting failure score.

Finally execute32 discarded native solver updates, B64/256 negatives, with a native linear dynamics model and the actual Offset1ModelMLP architecture. The source ignores its hidden_dim/num_layers constructor fields; preserve and disclose the actual layer sizes instead of claiming those fields set a new architecture. Use16 separate256-point toy trials, no latent labels, no validation/test selection, no formal checkpoint or paper metric. Verify finite losses/parameters, actual parameter change, no contrastive pair crossing trial boundaries, GPU execution, and unchanged source hashes. All32 updates are preflight-only and must never be added to formal reproduction/learning-comparison counts.

The immediate scientific decision still waits for the16 C54 models and their previously frozen scoring contract. Do not initiate a DCL full sweep or a new method solely because this preflight passes.
