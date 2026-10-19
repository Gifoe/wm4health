# Exploratory fixed-checkpoint readout-path controls

This plan is written after the fixed18-fit training/test job completed and after one wave-test progress row was seen,but before aggregate result analysis. It is a post-hoc mechanism diagnostic,not an independent confirmation or a new method. No model,hyperparameter,normalization or checkpoint selection changes.

Use all12 native-architecture checkpoints(two-stage32,two-stage4,4-noRevIN,end-to-end32,three seeds). Compute their mean predicted future latent at every lead and mean historical latent at every history position from TRAIN windows only. Future sensor targets are not inputs to these means.

At each held-out window,retain the native normalization chart and compare:(a)actual future latent,(b)replace only future latent by its fixed TRAIN mean while retaining the native history prefix,(c)replace both future and decoder-history latent tokens by their fixed TRAIN means. Keep the native decoder and all weights frozen. If replacing a path barely changes outputs/performance,it is evidence of weak functional dependence for this readout. Worsening under a replacement is not an identified causal effect and may include off-manifold distribution shift.

Also report the fixed64-step historical sensor mean as a zero-parameter forecast. The64 is the existing context length,not selected from test results. This is a direct reference for the information already present in native RevIN. Do not remove RevIN statistics without re-encoding and then mislabel chart mismatch as a dynamics failure.

Score all three previously fixed splits and every sensor/horizon;no selective system or seed reporting. Retain raw predictions,training-only mean latent arrays,model/prediction hashes and original-forecast replay error. The controls cannot establish that a latent has zero mutual information or that other readouts could not use it. The privileged future-target decode is already separately reported by the primary job.
