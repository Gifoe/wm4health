# C70: qualify the existing action-conditioned RP-GSSM contract

Frozen before execution. The C69 turn was progress: physical-task native simulation, independent replay, discrepancy attribution controls and a verified GitHub commit were completed. Overall research goal is still active/unachieved.

## Historical exclusion and purpose

RP-GSSM is not a new unexecuted baseline. C2 already trained its native Gaussian-site model and compared existing numerical-RPM/generative controls; that alarm/approximation route was closed. C46 already read action-conditioned/nonstationary source. Current author HEAD remains a75225d5edceb972bb872dc9a335608733567dd4. Do not repeat the old trained alarm experiment, declare a missing decoder/action prior, or infer a new framework from a software repair.

This is a component qualification of the previously source-only action-conditioned branch, not a trained model, original paper benchmark or research gap. Test what the existing source actually supports before considering uncertain-state/uncertain-system reuse as a lead. No neural optimization or latent-state learning occurs. The numerical reference has privileged known transitions and observation likelihood; it proves operator correctness only.

## Fixed matrix and data conditions

Use the unchanged pinned rp_ssm/actions.py, dists.py, dists_utils.py, rpm.py, distmaps.py and related source on server. Reuse C2's existing isolated .venv_rpm with CPU JAX and float64; no environment change and no unnecessary GPU training. All raw numerical arrays stay server G.

Latent dimension2, B4 sequences, T32. Seeds70001/70002/70003. Test continuous and discrete action mappers, additive and multiplicative transition matrices, and invariant N(0,I) versus explicitly noninvariant initial mean[.4,-.7] and covariance[[.9,.1],[.1,1.4]]:24 fixed fixture cells. Each has known stable base A, positive process noise and input-dependent transition/bias. Continuous uses the author's ActionNetwork around a small fixed randomly initialized Flax network, not a learned model; discrete uses two declared control choices. The observation likelihood is a fixed linear Gaussian sensor with finite varying precision, including weak-evidence time points. No model uses hidden latent labels for optimization; there is no optimization.

Compare (a) native to_chain mean/cov/adjacent cross-cov against independently constructed full Gaussian block covariance and recursive moments, and (b) native get_posterior versus dense Gaussian conditioning with the same known Gaussian sites. Explicitly account for the upstream smoother's1e-5 diagonal covariance padding: compare means and cov-minus-padding at1e-7; report the raw padding difference and adjacent-cross error separately. Cross-covariance raw difference may include the explicitly padded covariance; bound1e-4 is predeclared, not adjusted after results. Confirm the native batch moment_match's law-of-total-covariance calculation independently. This does not validate its approximation to an arbitrary empirical recognition mixture.

## Interface and metadata controls

Execute native discrete initialization as documented by ACLGParam.init; catch/record a signature failure, if present, before any bridge. A separate authored signature bridge may forward the explicit latent dimension/mode/bias arguments to the unchanged DiscreteChoiceActionMapper; label it an ordinary adapter.

For each fixture record whether native update(init_params) preserves initial-distribution flags and moments. If source drops a declared noninvariant flag, preserve the failed output and test a separate adapter that carries start_from_invariant and opt_params into the unchanged constructor. Do not overwrite upstream files or quietly call the repaired path unmodified native. The numerical contract test after a required adapter is labeled repaired-component qualification, not evidence of new representation or inference.

Additionally test a fixed continuous 'constant matrix / no action bias' configuration for action invariance. Record whether the source actually honors those flags. Do not use an unsupported/incorrectly bound path as a weak baseline in subsequent research. An explicit output-mask adapter is an ordinary config repair if required; its source and effect must be retained separately.

## Checks and decision

24 numerical cells, all retained, plus explicit interface/configuration checks. Save original errors, exact source hashes/revision, runtime, fixtures, native/reference arrays and machine-readable comparisons. Independently replay NumPy dense conditioning and law-of-total-covariance from retained arrays, and rerun one fixed native cell. No p values or performance superiority from numerical agreement. A failed software binding/flag path triggers an ordinary named repair, not an accepted research direction.

If the supported/fixed existing components match the Gaussian reference, reject the premise that action-conditioned recognition/smoothing is absent. This still does not prove that a learned reusable nonlinear world model, uncertain system parameters, or all observation models are solved. The next mechanism must survive the full prior-art and historical-control review; do not turn24 passing Gaussian fixtures into the final goal.
