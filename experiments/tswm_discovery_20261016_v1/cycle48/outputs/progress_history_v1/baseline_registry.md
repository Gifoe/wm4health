# C48 execution registry

UTC 2026-10-17T13:19:45.026138+00:00. Frozen16-fit queue, 4 completed fits / 9000 completed-fit updates at this snapshot. Qualification24 discarded updates excluded. Actual server supervisor pid41632; staterunning. Training/evaluation pending as of this timestamp; consult live supervisor/status files. No EVAL performance or new-method claim is available.

| Arm | Trainable / nominal parameters | Discarded CPU step seconds | Discarded CUDA step seconds |
|---|---:|---:|---:|
| node | 3329 / 3329 | 0.08750 | 0.21626 |
| fixed_dde | 3521 / 3523 | 0.15691 | 0.44494 |
| learned_dde | 3523 / 3523 | 0.21907 | 0.59119 |
| gru | 3585 / 3585 | 0.05181 | 0.04949 |

Article-linked model9b573c4bed5efee9fe9c6b765044dbd506b5bf07 and solver4888c63dc84e966f6c6553277adc3851a2cca4fc. Classes/solver execute unchanged; calendar wrapper, scalar residual GRU head, new data/task and training schedule disclosed in PROTOCOL.md. Cross-benchmark adaptation, not original-paper headline reproduction. No new proposed method.

CPU selected from actual discarded throughput; global package/environment untouched. Same DATA/seed minibatch schedule across four arms, fixed/learned delays share initial weights. GRU and complete-history ridge are the strong ordinary controls; NODE is a weak diagnostic ablation.

ett: TRAIN4017, VAL52; selected ridge100, VAL MSE0.0553560. These are VAL selections, not EVAL results.

jena: TRAIN2002, VAL21; selected ridge100, VAL MSE0.1058691. These are VAL selections, not EVAL results.

