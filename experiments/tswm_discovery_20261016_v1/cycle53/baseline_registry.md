# C53 baseline status

C53 snapshot UTC 2026-10-17T15:57:13.770701+00:00: **7/40 predictive fits complete**, 2790 completed formal optimizer updates; shared native AE complete. This is a dated running snapshot, not a finished result table. Second-batch model evaluation opened: False. Accepted final research directions: 0.

Two backbones (CrossLinear/PatchTST), four native design configurations and five seeds use the first original Greenhouse batch. Its815/163 train/VAL windows agree with the published163-window held-out specification. The second crop batch is a separate transfer audit after all checkpoints are sealed; it is not another independent real domain. The exact original paper's raw-byte run identity is not supplied, so headline numerical equality is not claimed.

Native upstream source:125 unchanged Python files, pinned before fitting. Observation concat, frozen-AE concat, frozen-AE Gate and native directional Gate are selected before outcomes; no test-dependent backbone selection. Native concat training/checkpoint-format equivalence qualified on TRAIN. Memory-limited PatchTST AE concat uses effective batch128/microbatch64 with a disclosed dropout/numerical adaptation. Persistence and validation-selected direct ridge are diagnostic controls in the frozen evaluation supplement.
