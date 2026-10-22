# Preserved preflight failure and simulator-continuation repair

The first executed worker(v2) terminated with exit1 before loading any model, at an exact-equality assertion on the unchanged portion of a middle-block edit. It had unnecessarily replayed that portion using stored float32 controls, whereas the originally generated simulator trajectory used the pre-storage float64 controls. The registered factual replay tolerance remains1e-5; no threshold or scientific criterion changes.

Worker v3 starts each branch at the original saved physical state immediately before that branch's first changed action, as in the existing C88 simulator protocol. Unedited prefix targets stay the original target array; identity still independently replays the whole future and must pass the original numerical tolerance. This avoids treating serialization roundoff as a causal response. No learner receives the simulator state.

The failed execution, log hash and original outputs/MATRIX remain untouched. Recovery has separate output/artifact/execution directories. It starts only after the recorded terminal exit1 and absence of the failed worker. No training, checkpoint selection, action permutation or metric is changed. The incident is an implementation error, not model failure.
