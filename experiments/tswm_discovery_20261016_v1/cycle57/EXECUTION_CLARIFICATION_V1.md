# Pre-training execution clarification

No formal fit has started when this clarification is authored. The native/CUDA qualifications and synthetic data generation are complete.

The proper-likelihood control has no stochastic network initialization, minibatch sampling or randomized optimizer. Repeating its identical initialization and deterministic full-data L-BFGS for all three neural seed labels would duplicate the same optimization. Execute two unique proper controls, one per sensor, with initialization label 2021; do not claim three independent seeds or report zero seed variability for it. The five neural controls still run all three seeds on both sensors: 30 neural fits plus two proper fits, 32 unique fits total. All maps and models are sealed before held-out access as in PROTOCOL.md.

Use float32 CUDA for the neural controls as qualified; use float64 CUDA for the small proper-likelihood model so L-BFGS termination is not dominated by float32 likelihood rounding. Runtime, precision, objective and budget remain explicit. This does not change data, neural architecture, gradients, seed list, update counts, endpoints or selection rules. It removes redundant deterministic work.
