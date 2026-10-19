# Baselines and execution scope

| Object | Execution | Role and limitations |
|---|---|---|
| DynaMix Python39cc55514a4c5a0c50534c092fd9c3d40abdabc8, released3D model | cycle18 nine native10000-step draws; retained here | Frozen10256-parameter autonomous model; three public examples, not paper54-system aggregate |
| Historical context replay | Retained; scored on path events | Descriptive memory control, not a learned universal model |
| Eight-neighbor successor resampling | Three seeds on each original context | Old analogue idea; strict past-only successors; not DynaBase or an exhaustive strong baseline |
| Native attention sigma=0 | Three original systems, one deterministic draw each | In-memory ablation; all other checkpoint tensors/disk bytes unchanged |
| Common-origin Chua native history/normalization wrappers | Three conditions x three stochastic draws | Same future and reference thresholds; more history is additional information, not a fair method win |
| Dense ridge vector-field library | Five past-only fits,12 validation candidates each;10000-step autonomous rollouts | Classical per-system fit, degree2/3 polynomial plus optional decile hinges, four-substep RK4; no known plant equations; different training budget from pretrained model |
| DynaBase/VAMPnets/transformer transition-rate method/Jacobian training | Literature only | Their absence from execution prevents a claim that existing methods are insufficient |

Runtime: native/model-control jobs use server E:/Anaconda/python.exe3.13.9,torch2.11.0+cu128,numpy2.3.5,CPU,process-local single thread. The retained event audit and its index verifier use the existing project .venv Python3.10 with numpy1.26.4; command paths are captured per job. RTX5090 was not needed for these short inference/control jobs. No package installation or mutation of either environment. Ordinary regression is new fitting; there is no new neural-network training.

Scientific job statuses, stdout, stderr and exact entry-script snapshots are under outputs/jobs. Plans distinguish original statistics, post-failure all-offset diagnostics, and later exploratory attribution/control fits. No public publishing or repository push.
