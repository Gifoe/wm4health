# Existing native EM option: matched learning control

Declare before any C54 heldout score is opened. This is an additional established-source-option control, not a proposed method. The original 12-fit C54 comparison remains intact.

Run the unchanged native RP trainer with `Config.em=True` on both original tasks and seeds 0,1 (four fits). Keep the original native generator/data files, A=.5I initialization, true latent dimensions, linear constant-diagonal recognition, Adam .001, batch32, singular-value clipping, precision, and 10,000-update terminal checkpoint rule. The only scientific configuration difference from the corresponding original RP fits is stopping the posterior gradient in the M-step. Compare native initial parameter arrays exactly against the False initialization for each task/seed.

All four fits use training observations alone. They use the existing CPU runtime on logical processors24-31 while the original queue uses16-23. Record wall times separately; CPU groups are not a controlled equal-time measurement. No installed package or native numerical source is edited.

The prior frozen-gradient diagnostic does not establish good trained-model quality. This control tests whether the already-provided source option improves the learned transition law and causal forecasts, before designing a new optimizer or world-model architecture.

Seal all four terminal models. The amended postprocessing gate waits for both the original12 and these4 model seals. Fit and seal all four extra training-only affine state/observation diagnostic maps first, then execute the original12-map preparation and heldout scoring, followed by the extra4 scoring. Consequently all16 models and all16 diagnostic maps are fixed before any heldout result can influence a choice.

Use exactly the same observation prefix50, horizons0,1,5,10,25,50, state/observation MSE, transition eigenvalue error, and discrepancy from the true-parameter Kalman conditional mean. Pair True/False within task and initialization for the four matched comparisons. The remaining original False seed2 is not silently dropped from its original three-seed aggregates. There are only two synthetic tasks, one data draw each, and two matched initialization pairs per task; no real-domain or significance claim follows.

No new hyperparameter, EMA estimator, negative-sampling rule, checkpoint selection, architecture, or test-driven tuning is introduced. If this source option repairs the effect, reject the claim that the current gradient mechanism itself establishes a missing world-model primitive. If it does not, distinguish encoder/reference approximation, fixed-posterior inference, and actual transition-quality evidence before proposing a method.
