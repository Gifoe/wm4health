# C53 live state

C53 snapshot UTC 2026-10-17T15:57:13.770701+00:00: **7/40 predictive fits complete**, 2790 completed formal optimizer updates; shared native AE complete. This is a dated running snapshot, not a finished result table. Second-batch model evaluation opened: False. Accepted final research directions: 0.

Two backbones (CrossLinear/PatchTST), four native design configurations and five seeds use the first original Greenhouse batch. Its815/163 train/VAL windows agree with the published163-window held-out specification. The second crop batch is a separate transfer audit after all checkpoints are sealed; it is not another independent real domain. The exact original paper's raw-byte run identity is not supplied, so headline numerical equality is not claimed.

Active training source: train_native53_v6.py; capture_native53_v6.py; selection SELECTION_SEAL_V6.json. Evaluation source: score_native53_v7.py; verifier: verify_native53.py. The old V1 and V5 supervisor states are historical terminal/controlled-repair traces and must not be used to infer a live worker. Inspect actual process identity before restarting anything.

Raw data, caches and checkpoints remain in the user-authorized large-disk project directory. No result synchronization to the workstation. Public target: Gifoe/wm4health.
