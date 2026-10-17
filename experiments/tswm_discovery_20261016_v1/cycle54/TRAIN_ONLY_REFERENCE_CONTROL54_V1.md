# Balanced target / fixed reference control

This follow-up responds to the completed training-only objective audit. It does not amend the formal training/heldout protocol. Use the same two completed RP checkpoints and training observations only.

The first audit found opposite A-scaling gradient signs for batch-32 and all-200 objectives. To distinguish reference-bank effects from unequal averaging of target sequences, form 25 batches of 32 target indices from four fixed-seed permutations of all 200 training sequences, concatenated before reshaping. Every training sequence occurs exactly four times. Freeze seed 5425 before executing the control.

Compare three quantities at identical frozen parameters: the native all-200 objective gradient; a native-objective per-target decomposition retaining all 200 reference sequences while averaging the balanced target batches; and the native batch-32 objective on those exact same balanced target batches with their own batch references. The first two must agree after averaging, to numerical tolerance. The latter can differ only through reference/normalizer composition and associated encoder derivatives, because target frequencies are equal and each posterior is computed independently by sequence.

Verify the decomposed all-target loss and A gradient against the unchanged native objective before interpreting the control. Retain all 25 gradients from both reference conditions. Report A-scaling derivatives and gradient norms/cosines. All original checkpoint/source hashes must remain unchanged; no optimizer step, label read, heldout read, checkpoint choice or formal training change is allowed.

This is an attribution control of a known contrastive-style finite-reference issue. The complete-reference auxiliary objective remains an approximation to proper RPM likelihood. A direction change under this control cannot establish that it improves state recovery, that the full-reference objective is statistically correct, or that a new method is required.
