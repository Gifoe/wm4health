# Existing hidden-state inference: qualification and novelty limit

The existing-capability check is closed. This is not a new world-model method, a held-out performance result, or a reproduction of the original hidden-node table. The formal three-seed training and frozen free-simulation assessment remain unchanged and incomplete. No final research direction is accepted.

## What actually ran

The pinned modular-metalearning source already offers hidden-node inference. Section 5.2 of the paper explicitly optimizes an unseen particle's initial state through learned dynamics, and first trains its demonstration with true graph labels: https://arxiv.org/html/2310.07015v1 . The current unsupervised-library checkpoint is therefore a protocol transfer, not that original experiment.

The first formal seed's closed epoch1000 checkpoint was fixed by epoch before this check. All computations ran on CPU in a separate process; shared environment and the training process were unchanged. Device-only source adapters replace CUDA defaults/allocations. The exact native batched_evaluate AST body contains no device replacements. Independent float64 NumPy message passing agrees with the adapted native 48-step rollout on two TRAIN rows to max absolute errors1.2619044559292192e-7 and1.894063427210213e-7. Source, checkpoint and raw artifact hashes are retained.

## Pinned implementation dependency, not a claimed paper invalidation

The native hidden branch constructs its current state from structure-specific parameter indices, but constructs the candidate from the beginning of the global parameter list. Its candidate comparison uses pred[1:] and TrainOutput[1:], which slice time, retaining the supposedly hidden node target. A constructed49-task fixture set current parameter indices49..97 and candidate indices0..48, kept observed targets at the same midpoint of the two native predictions, and varied only the hidden targets between those predictions. All49 candidate decisions changed. This executes the unmodified native comparison and its subsequent bookkeeping; it demonstrates a hidden-target dependency for this contract. It does not show that a particular reported paper run used this parameter ordering or establish its reported numeric result is wrong. It is an implementation/protocol issue, not the requested research gap.

## Ordinary observer actually works as an execution baseline

An explicitly separate observer uses the native frozen composer and stored TRAIN graph assignments, four initial coordinates per hidden node, zero initialization and100 Adam updates at lr.01. Only the first25 time points of TRAIN rows0..3 and nodes1..4 supply the objective; node0 truth never supplies optimization targets. Replacing every hidden input/target value from-123 to456 leaves all101 objective values,100 hidden-gradient arrays, final estimates and Adam states identical. Hidden-target gradients are exactlyzero; initial-state gradients are nonzero; module gradients remain absent and all library/checkpoint hashes remain unchanged.

Mean visible-history reconstruction MSE changes from0.0006915383273735642 to0.000008196262569981627. All four individual before/after values and the full loss trace are retained, with no task filtering or tuning. This is fitting previously pretrained TRAIN trajectories, not future prediction, hidden-state accuracy, state identifiability, an independent held-out test or a capability-gain claim. It establishes that the ordinary control is executable and information-safe for later tests. It does not prove that this baseline solves every observation shift.

## Preserved engineering failures

V1 failed before the source branch because our adapter retained the data's float64 dtype while the loaded native modules are float32. V2 corrected that conversion but its independent NumPy reference omitted the edge module's final ReLU; the mismatch gate stopped execution. V3 corrected the independent reference and closed successfully in18.8764773 seconds. Both failed sources/logs/terminal receipts remain; neither failure is scientific evidence of a model defect. No change to the model's mathematical operations was used to make the reference pass.

## Research consequence and next decision

Do not promote generic initial-state optimization, alternating state/graph search, or frozen-dynamics observation fitting as novel. These overlap directly with the source and existing observer/structured-state-space methods. CandidateA still requires a distinct residual after ordinary controls on independently held-out observation/mechanism combinations. The existing C71 cohort also cannot support unseen-topology claims: its TRAIN graphs cover all34 five-node undirected graph isomorphism classes. Preserve that limitation rather than redefining its split after training.

The next decisive result is the already frozen three-seed common-history future rollout with graph-inference and ordinary history controls. Continue the original fit budget; keep reserved EVAL closed. Only after that closes should a new observation/structure generalization cohort be specified prospectively, with complete observers as controls. C72 remains a separate closed information-contract audit, not a new method. Overall goal ACTIVE/UNACHIEVED; accepted final directions0.
