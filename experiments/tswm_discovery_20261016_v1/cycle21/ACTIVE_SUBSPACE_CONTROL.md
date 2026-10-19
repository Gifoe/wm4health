# Independent-cohort mature-compressor control

Fixed while the120-student main study is still training and before its test results. This extends the cycle20 classical control to all four cycle21 families and three independently fitted teachers per family. It does not change the120-fit source, losses, checkpoint or selection rules.

Run only after the main scientific job terminates successfully and its independent verification passes. Reuse its train/validation/test splits and teacher checkpoints. All-output versus action-response-output sensitivity reduction uses the same algorithm as cycle20:95% train response energy output span,512 training states and four supported actions for input Jacobians,top4 input directions, train-only whitening, polynomial degrees1/2/3 and ridge1e-6/1e-3/.1 selected by validation latent MSE. Labels of the physical state/gain are forbidden for these choices. Only the final frozen diagnostic linear decoder receives those labels.

This has24 fits. Queueing is a separate clipped-response stress test, not a pure replication of gain modulation: some action effects are physically equivalent at the boundary. Do not hide negative results or retune degrees/rank/dimension using test metrics.

Compare all-output and action-output versions to distinguish output-target choice from the established active-subspace compressor. The action-output variant shares the candidate target-selection ingredient, so its success cannot establish whether that ingredient is new. It can disprove the need for a new neural compressor. It is independent of cycle20 trajectories/rotations/teacher weights, but not a realistic source-native dataset or a multi-step simulation test.

Save coefficients,feature powers/scales,transform,decoder,predictions and hashes. Verify by direct monomial products rather than sklearn's feature transformation; independently recompute metrics,whitening,basis orthogonality,validation choice and analytical physical response. No original experiment or public content is modified.
