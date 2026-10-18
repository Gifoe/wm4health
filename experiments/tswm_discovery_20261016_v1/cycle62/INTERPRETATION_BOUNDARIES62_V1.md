# Interpretation boundaries fixed before test opening

Informer has47527 parameters; PRO and DYN each52558 in this configuration. Hyperparameters follow the authors' corresponding scripts, not an exact equal-parameter/equal-FLOP constraint. Comparisons identify these source configurations, not an isolated causal effect of block placement or model size. The two modified arms have equal parameter counts but different computational paths.

Four fits use original per-window data copies, five use the qualified cached tensor copies after runtime faults. Their tensor values and gradient calculation were checked, but their wall-clock times should not be compared as architecture-efficiency evidence. Completed-fit training updates and known lower bounds for two aborted attempts are separate. No training runtime ranking or finite-budget superiority claim will be inferred here.

All120 metric rows are reported; there is no selection of the favorable seed, inference RNG, origin subset or evaluation horizon. Three inference streams are not Bayesian trajectories. Test observations can enter a later rolling origin's history, as in normal rolling forecasting, but never enter that origin's free future rollout. Common-origin96/192/384 scores share start states; full stride1 official96 has a different origin population.

This measured series has no paired counterfactual transitions or known closed autonomous physical law. A ranking reversal or long-horizon MSE increase alone is not a novel mechanism gap, and no method will be tuned on this screen's test predictions.
