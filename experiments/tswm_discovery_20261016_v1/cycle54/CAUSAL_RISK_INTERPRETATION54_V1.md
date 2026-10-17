# Prescoring interpretation of the existing16 frozen fits

This adds read-only analysis after the previously registered scoring pipeline completes. It does not change any model, learning budget, selection, diagnostic map, split, horizon or existing report. All16 models/maps and the existing two evaluations must be sealed before any artifact is read.

For each test sequence and every prespecified horizon0/1/5/10/25/50, let p be the learned physical-coordinate state mean, m the true-parameter causal Kalman conditional mean and z the realized state. Independently recompute:

    factual = mean((p-z)^2)
    oracle_noise = mean((m-z)^2)
    excess_conditional = mean((p-m)^2)
    finite_sample_cross = 2*mean((p-m)*(m-z))

The exact sample identity is factual = oracle_noise + excess_conditional + finite_sample_cross. The cross term vanishes in conditional expectation, not necessarily in these50 realized sequences. A finite realized MSE below the oracle's realized MSE is therefore not proof of beating the Bayes predictor. Conditional-mean discrepancy is already in the main protocol; this analysis supplies its full risk interpretation.

Also report the mean predictable-state energy mean((m-unconditional_mean)^2) and excess_conditional divided by that energy. Propagate the generator's actual m1/A/b to obtain each unconditional mean. Report both raw and normalized errors at all horizons. A zero/underflow denominator gives an undefined ratio, not zero error. Large long-horizon normalized errors can coexist with small absolute differences when predictable energy is nearly gone; do not select a favorable horizon or claim a new metric.

Uncertainty: use2048 sequence-cluster bootstrap replicates, fixed NumPy seed5491, common resampled indices across methods and horizons within each task. Give pointwise95% percentile intervals for every comparison, not multiplicity-adjusted significance. The unit is the complete heldout sequence, not individual time points. These intervals condition on the frozen training data/models and do not cover initialization uncertainty, independent data-generating mechanisms or real-domain generalization.

Comparisons: retain the4 exact True-vs-False initialization pairs. Within each task, also compare mean per-model error of three False fits versus three ordinary corrected EM fits, and two True fits versus the corresponding two EM initialization labels. The latter are fixed-model group comparisons on common cases, not new ensembles or claims of equivalent training initializations. No model or hyperparameter is selected from these intervals.

Decision boundaries: a competent existing-source setting or ordinary EM remedy weakens a missing-capability claim; a residual error from one model on two in-model linear tasks does not establish a new framework gap. Accurate long-horizon prior means alone do not prove faithful dynamics. Any new method still needs an independently motivated setting, strong controls and fresh heldout validation. This analysis accepts no new research direction automatically.

Qualify the arithmetic only with independent synthetic fixtures: exact finite-sample identity; symmetric Gaussian cubature with analytically zero cross term and known trace covariance; constant paired differences; undefined zero-energy ratios; explicit sorted/interpolated quantiles versus NumPy. These fixtures access no formal model, data or labels. Record qualification and registration hashes before the first heldout score. Run automatically only after postprocess54_v3 completes; leave all pre-existing reports/manifests untouched.
