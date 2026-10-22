# Execution repairs, retained separately from scientific results

The first adapter attempt failed its exact text-replacement guard before model initialization: one path expression occurred once, while the adapter required twice. All six zero-update failures remain recorded. V2 fixes the two distinct patch-path expressions and writes a byte-exact canonical diff.

The second controller then collided with its predecessor's unversioned per-cell launch filename. Its first worker had already been created and completed the H336/seed42 fit. Direct process inspection confirmed that the worker was terminal; its final metrics, unchanged-source receipt, checkpoint and prediction hashes and final completion log survived. The forecast checkpoint **and prediction NPZ are byte-identical to the old original seed42 run**. This is real repeatability evidence, not a favorable seed selection.

V3 isolates controller receipts under a new directory, reserves each new launch receipt before process creation, and recovers that completed first worker without training it again. The absence of a process-exit receipt for that worker is explicitly recorded; it is not silently replaced by a fabricated exit code. The remaining five fixed matrix cells use the same V2 scientific worker.

No original paper code, original baseline directory, registered seed/horizon set, optimizer/loss, selection rule or Test hyperparameter was changed to fix these execution issues. They are engineering mistakes in this campaign, not model failure cases.
