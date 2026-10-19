# Experimental results

Generated UTC: 2026-10-16T00:04:19.569192+00:00

All claims below refer to completed remote execution. Distinct protocols are not combined into a leaderboard.

## Full native A configuration

- Complete:10 epochs,15470 updates,462921 parameters,1029.4s.
- Best validation Spearman permutation MCC:0.840796941,epoch1.
- End-of-training MCC:0.480325728;the best checkpoint remains preserved.
- Validation latent labels select the checkpoint, matching the synthetic validation convention. No independent test or paper-exact replication is claimed.
- Reconstruction is stochastic; rechecking the same best model can change reconstruction MSE while posterior-mean MCC stays fixed.

## Native real-world parameter registration

| Seed | Prior parameter tensors | Registered in full model | Max prior weight change after native optimizer step | Eval change after prior zeroing |
|---|---:|---:|---:|---:|
|81|40|0|0.0|0.0|
|82|40|0|0.0|0.0|
|83|40|0|0.0|0.0|

Nonzero prior gradients exist, but the native optimizer does not contain those tensors. Evaluation reconstructs supplied windows; this is not evidence of blind prior rollout. This is an implementation/protocol result, not a theorem refutation.

## Generator audit

Original innovations:rank4,maximum off-diagonal absolute correlation1.000000000.
Independent-noise control:rank7,maximum off-diagonal absolute correlation0.002656642.
The pinned helper named leaky_ReLU returns its input unchanged. Active transitions/mixing therefore use linear operations. The hierarchy reuses noise indices across layers. Fixing noise indexing does not establish sufficient variability, invertibility, independent-view or every other theorem premise.

## Supervised recoverability certificates

| Dataset | Context width | Validation mean Spearman to true factors | Validation mean R2 |
|---|---:|---:|---:|
|native_A|1|0.995928647|0.993061163|
|native_A|3|0.999236957|0.998706461|
|native_A|5|0.999502463|0.999176061|
|independent_noise_A|1|0.957329156|0.928324397|
|independent_noise_A|3|0.962544657|0.937362782|
|independent_noise_A|5|0.963243632|0.938904943|

The certificate uses factor labels during fitting. It shows signal accessibility for these finite data, not unsupervised identifiability or a superior fair model. Independent innovations leave nonzero residual error for hidden factors at finite context.

## Conditional-density isolation

| Model | Seed | Selected step | Mean proper test NLL over3 fixed contexts | Maximum scalar mass error |
|---|---:|---:|---:|---:|
|native_inverse|81|2000|1.234863162|9.99e-16|
|normalized_gaussian|81|1800|0.925280352|0|
|native_inverse|82|2000|1.215042392|4.44e-16|
|normalized_gaussian|82|2000|0.924777786|0|
|native_inverse|83|2000|1.090493560|2.22e-16|
|normalized_gaussian|83|2000|0.924177587|0|

All9 trained native scalar conditionals tested are monotonic and normalized to numerical precision. The ordinary normalized Gaussian fits the Gaussian DGP better at this fixed budget. That is an existing control, not a new proposed method. Three model initializations share a fixed data cohort; they are not3 independent datasets.

## Native trained-prior conditional diagnostics

| Checkpoint | Scalar conditionals | Maximum mass error | Minimum reachable base-noise mass | Conditionals with derivative sign changes |
|---|---:|---:|---:|---:|
|initial|21|8.74314659e-07|0.49999951|18|
|best|21|0.00435048454|0.501893404|3|
|final|21|5.55111512e-16|1|0|

These are3 fixed training trajectories ×7 scalar factors, not held-out model selection or a global joint certificate. A folded inverse can have an integral near1 while failing to cover half the base Gaussian; normalization and invertibility are distinct. The final prior has normalized, unfolded tested conditionals. The generic claim of systematic trained likelihood cheating is unsupported.
