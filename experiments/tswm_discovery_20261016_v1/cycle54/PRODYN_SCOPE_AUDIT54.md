# PRO-DYN source/task scope audit

Checked against arXiv v3 (14 May 2026) and current public commit `733d986d349e9e71ded5d9cdeeeed8d3323d83b4` (20 May 2026). C3 had read v2; the public code pin remains current.

The paper explicitly studies its first family of forecasting architectures. Its DYN component maps past observations/features into a forecast block. The public variants include Informer, FEDformer, FiLM, MICN, PatchTST, iTransformer and Crossformer. For example, `PatchTST_post_DYN` adds a learned time-axis linear mapping before patching/encoding; `Informer_DYN` uses a learned mapping to initialize the forecast portion of its decoder. The reported tasks are conventional forecasting datasets through the TFB adapter. This is useful evidence about predictor parameterization. It is not a released latent Gaussian inference model or a controlled hidden-state benchmark.

Sources: [v3 paper](https://arxiv.org/html/2507.15774v3), [pinned repository](https://github.com/ARBrachet/PRO-DYN/tree/733d986d349e9e71ded5d9cdeeeed8d3323d83b4), source files `ts_benchmark/baselines/time_series_library/models/PatchTST_post_DYN.py` and `Informer_DYN.py`.

Our inference is that a native PRO-DYN forecasting run would not answer the current RP-GSSM question about posterior-state recovery versus learned transition quality. A new Lorenz/hidden-state migration would be our experiment, not a native experiment supplied by this repository. Do not invent such a released task or count source review as reproduction. The pending PRO-DYN intake mentioned by the original C54 report generator is now resolved by this note: retain it as a forecasting architecture reference, and do not launch it merely to add another dynamics-named baseline.

This does not exclude PRO-DYN from a later same-information forecasting comparison. It also does not show that an ordinary model solving a linear task makes every nonlinear representation-learning problem solved. The research standard remains a useful, reproducible advantage over competent same-information controls on a meaningful task. It does not require proving that every existing method is fundamentally incapable, nor does using known components by itself defeat a possible contribution.

Current decision: no extra PRO-DYN training justified by this audit. Complete the predeclared C54 comparison before attributing its errors. Any next nonlinear or real-system experiment must be selected from its actual released input/task contract and distinguish tractable posterior arithmetic from model/learning quality. No new research direction is claimed here.
