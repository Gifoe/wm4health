# Cycle16 research decision: close the covariance-repair method line

**Research Question:** Does loss of persistent-context dependence in a hierarchical latent world model create a consequential failure after end-to-end learning that requires a new method?

**Evidence:** Native MTS3 was trained1500 updates on a fixed, verified history/calendar protocol. The selected step1100 has validation NLL0.231094,MAE0.209789,coverage96.502%. Replacing its covariance recursion with an ordinary full joint recursion at the same mean gives NLL0.240164,coverage97.101%. Independent closed-form verification covers all69 origins and75 horizons. The channel matters, but no complete-model undercoverage failure or NLL benefit is established.

**Existing Methods:** Ordinary joint Gaussian propagation corrects the earlier admissible scalar witness cheaply. Persistent model samples, functional neural processes, latent-parameter inference, RAMP, probabilistic meta-system-ID and FIM-SDE cover the generic uncertainty/dynamics ideas. A specialized covariance architecture is unjustified here.

**Proposed Idea:** None accepted. Stop covariance repair/compression as the present new-method route. Do not convert a source approximation into a paper claim when the learned model does not display the asserted failure. A future general time-series world-model framework needs a different, consequential residual beyond these existing methods.

**Falsification:** The tested claim that a simple joint-covariance replacement improves the frozen learned model is unsupported. This does not prove all corrected retraining or all reusable-belief questions solved. Initial cross-covariance is unknown, and the post-hoc variance decoder may be out of distribution. One validation-selected seed cannot establish a universal negative result.

**Feasibility:** The full source-native model, checkpoints, optimizer/RNG state, data versions, raw logs and diagnostic arrays are preserved on the RTX5090 server. This run is a transfer pilot, not original-paper reproduction. The test segment remains untouched. Verified selected weights and36 scientific binary files have been mirrored locally.

**Next Experiment:** The available pretrained probabilistic meta-system-ID checkpoint is now verified and its native interface preflight passed, without new fitting. Evaluate its source-defined cross-system prediction protocol, reconciling normalization and future-information boundaries before any prospective simulation claim. Freeze a minimal held-out test separating marginal forecast calibration, joint trajectory simulation and evidence updates, with strong ordinary covariance/functional/generative controls. Abandon the new gap if these controls already suffice; do not rename them as an invention.

Final recommendation: none yet. Accepted novel directions:0. The overall autonomous research objective remains active and unachieved. This is a completed cycle decision, not a completed research project.
