# C49 pre-EVAL matched-cost comparison rule

This supplement does not alter the frozen training data, objectives, seeds, optimization, primary checkpoint selection or original full evaluation. Its upload/seal receipt must establish that the rule was recorded before primary EVAL opening. No EVAL score enters any choice below.

Question: do any quality differences survive a common training-and-validation time budget, rather than just a common maximum update count?

Budgets are fixed at60,120,180 seconds per fit. For each case/seed/arm/budget, consider only validation checkpoints whose measured `elapsed_seconds` is at most that budget. Select the checkpoint with lowest observed VAL kernel score among those available; break ties by earlier step. If a fit ended before the budget, it contributes its best available checkpoint. If none exists, mark unavailable rather than substituting a later checkpoint. Validation and data-processing costs inside the measured fit clock are included; preflight, process launch and data setup are excluded and remain disclosed.

Evaluate all available selections on the unchanged EVAL episodes/density contexts with128 iid futures and the frozen physical energy/MSE definitions, open-loop24-bin horizon only. Report all budget points. Do not select a preferred time budget or hyperparameter from EVAL results. Checkpoint duplicates may reuse the exact same evaluation bytes identified by their SHA. A primary selected checkpoint is reusable only when its hash is identical.

New predictions must retain their actual innovations and undergo the same independent NumPy model/energy/MSE replay. Compare paired physical episodes and keep density views together. This frontier does not replace the complete primary observed-law/reveal evaluation and does not establish real-domain utility or novelty. It can be stopped as a documented unexecuted extension if the primary recipe is clearly negative and the additional compute would not change the research decision; missing points must never be represented as completed results.
