# Frozen normalizer-estimator attribution

The balanced-target control established reference-composition sensitivity at two frozen checkpoints. This next diagnostic separates forced self inclusion from random-normalizer estimation, without optimizing parameters or reading labels/heldout observations.

At each of the same two checkpoints, cache the full native reference-score matrix G[t,i,j] and its directional derivative under A -> alpha A at alpha=1, plus the per-target auxiliary-loss derivative. Verify the mean derivative against the already computed native autodiff result. Verify cached reconstruction of all 25 balanced native batch derivatives against the earlier direct native gradients. Both are prerequisites for interpretation.

Use RNG seed 5426, 512 burn-in and 1,024 recorded batches, target size32 and reference size32, uniformly sampled with replacement from the fixed 200 training sequences. Compare all of:

1. Native coupled reference: each target batch is its own reference bank.
2. Leave designated diagonal out: use the other 31 batch entries as reference for each target; duplicate recording identities at other entries remain. For a target conditioned on its draw, the remaining entries are independent uniform draws.
3. Independent reference batch of 32 sequences.
4. Independent reference numerator with the exact frozen all-200 normalizer. This is an oracle-normalizer control requiring the full training bank; it is not a practical training method or hidden-state oracle.
5. Two running-normalizer estimators on independent references, gamma=.9 (the inspected SogCLR source default) and .1. Each maintains a log-space running mean per training sequence and time; initialize a new sequence's normalizer at its first sample mean and update once per unique target per batch. Normalizer values are detached in the gradient coefficient. All variants are reported, without choosing a future training hyperparameter from these values.

The running-mean controls adapt an established stochastic-compositional optimization principle; they are not executions of the official image SogCLR baseline. Their behavior at frozen parameters does not establish tracking accuracy while model parameters change. For the independent controls, report Monte Carlo SE over the 1,024 independent recorded batches. For temporally correlated running means, report mean and standard deviation only, without an iid confidence interval. This is conditional MC uncertainty at two fixed models/data, not across data draws or systems.

Retain score/derivative caches and per-batch estimates on the large server disk. No optimizer updates, checkpoint changes, state/readout fitting, heldout access, or formal-protocol changes. A simple established control fixing the frozen estimator defeats a claim that this specific gradient issue alone requires a new world-model architecture. State recovery, rollout quality and remaining approximation error must still be tested separately.

Prior methods: [SogCLR, ICML2022](https://proceedings.mlr.press/v162/yuan22b.html), [gradient-bias study, NeurIPS2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/db174d373133dcc6bf83bc98e4b681f8-Abstract-Conference.html). Inspected source: Optimization-AI/SogCLR commit `ef289b52ed72d22d8b3c4e8a76d698c117f02a44`, `sogclr/builder.py`.
