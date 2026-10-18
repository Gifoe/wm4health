# Native frozen-model future interface, before execution

Purpose: qualify a genuine common-history future simulator, distinct from the author's reset-block development metric. No new method or performance claim. C71's running fits, native-blockVAL checkpoint selection and sealedEVAL remain intact.

Use the immutable first-seed epoch50 checkpoint from the already declared periodic grid, with its recorded SHA. It is a development checkpoint, not the selected final baseline. Strict-load all module and learned-proposal parameters/buffers. Qualify on original TRAIN rows0,1,2,3 only. No VAL/EVAL forecasting error is computed.

For each49-step history, initialize a native five-node graph structure randomly, then apply100 unchanged native proposal/acceptance updates using only those49observations. The pseudo dataset duplicates the history into the source Train/Val fields to avoid exposing a future target; dummy graph labels enter diagnostics only. The inference temperature schedule follows the source adaptive formula with its initial-9.8, acceptance-1, slope800 and1.1factor. No weight update, graph-truth supervision, or validation hyperparameter search.

After inference, keep the graph fixed and use the source composer with self_regress_steps=H forH1/10/50. Seed it only with the last observed state. Supply a paddedH+1array; compare all scored predictions with future padding0 versus111. The extra source batch induced by padding is outside the scored branch. Require finite scored outputs, matching shapes and exact padding invariance. Hash model states/checkpoint/source before and after.

For the first two graphs, independently evaluate the actual learned neural modules in NumPyfloat64 and manually sum edge messages/update nodes for50steps. This separate calculation must agree with the nativefloat32 prediction with max absolute error divided bymax(1,maxabs(reference)) <=2e-4. Retain raw arrays, per-horizon diagnostics, fitted graph assignments and hashes on serverG. This is numeric/interface qualification, not a learned-model accuracy test.

Later common-history performance assessment requires a separate frozen TRAIN/VAL protocol, complete same-information controls and final selected checkpoints. EVAL remains closed during failure discovery. A discrepancy on this still-developing checkpoint would not establish a research gap or a paper-level failure.
