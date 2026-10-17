# C54: Jointly learned RP-GSSM linear audit

Status: all 12 fixed-budget fits (RP native; ordinary LGSSM EM with the explicitly qualified initial-covariance normalization correction), training-only state maps, causal heldout scoring and independent numerical checks completed. This is a paper-recipe reconstruction using pinned native components; it is not the original full hyperparameter sweep or visual benchmark reproduction.

No new research direction is accepted from these two synthetic tasks. A per-model error does not establish a field-wide deficiency, and an already available LGSSM repair does not justify renaming it as a new world model. High state-decoding R2 alone cannot establish accurate propagation; all forecast and conditional-mean results must be considered. Low factual long-horizon R2 alone cannot establish failure because the oracle itself loses predictability.

Execution history is preserved: first Start-Process launch exited before preparation with empty logs; persistent WMI preparation initially failed because the remote staging lacked the two pinned demo notebooks; filling only those missing files made all 21 source hashes match. No previous source file was replaced. The first inference qualification exposed the source's explicit 1e-5 covariance correction; the versioned evaluator checks that term explicitly. The original ordinary EM M-step also omitted a sample-count normalization in its initial covariance. It produced NaNs after the first update; the standard moment fix was independently qualified before completing baseline fits. These are implementation/checking issues, not a new model failure mechanism. No formal model/objective/hyperparameter changes were made to obtain favorable heldout scores.

| Task | Model | h=1 conditional-mean discrepancy | h=10 discrepancy | h=50 discrepancy |
|---|---|---:|---:|---:|
|linear_small|rp|0.0322689|0.0715555|0.073247|
|linear_small|em|3.4704e-05|3.23437e-05|2.00214e-05|
|linear_medium|rp|0.248114|0.199638|0.0972055|
|linear_medium|em|0.000188373|0.000844449|0.000463326|
