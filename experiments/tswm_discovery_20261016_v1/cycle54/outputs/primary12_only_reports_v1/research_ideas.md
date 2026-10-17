# C54: Jointly learned RP-GSSM linear audit

Status: all 12 fixed-budget fits (RP native; ordinary LGSSM EM with the explicitly qualified initial-covariance normalization correction), training-only state maps, causal heldout scoring and independent numerical checks completed. This is a paper-recipe reconstruction using pinned native components; it is not the original full hyperparameter sweep or visual benchmark reproduction.

No publication-ready idea is recommended. The candidate question is whether state recovery and propagation quality separate under complete unsupervised learning, after controlling information, uncertainty and optimization. This is an audit question already related to extensive state-space literature, not an innovation claim. A new method is unjustified until an existing strong model fails in a scientifically meaningful setting and routine remedies are excluded. C54 supplies a stronger executable reference and a conditional-noise-aware diagnosis for that decision.

| Task | Model | h=1 conditional-mean discrepancy | h=10 discrepancy | h=50 discrepancy |
|---|---|---:|---:|---:|
|linear_small|rp|0.0322689|0.0715555|0.073247|
|linear_small|em|3.4704e-05|3.23437e-05|2.00214e-05|
|linear_medium|rp|0.248114|0.199638|0.0972055|
|linear_medium|em|0.000188373|0.000844449|0.000463326|
