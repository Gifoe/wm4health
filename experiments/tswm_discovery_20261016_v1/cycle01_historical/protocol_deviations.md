# Protocol discrepancies and amendments

The original config and plans are retained, with their original SHA256 hashes.
These notes do not retroactively edit them.

1. **Scalar WorldTS screen checkpoint selection.** `config.json` describes
   selection as factual-validation MSE only. The executed native two-stage
   script instead selects **stage 1 by its native latent validation loss**;
   stage 2 and all end-to-end/FIR/GRU final models use factual validation MSE.
   This is visible in `run_synthetic_screen.py`, which was the executed source.
   No discovery/confirmation labels select checkpoints. Do not describe all
   training stages as having identical selection objectives. The structural
   support zeros and the same-core end-to-end control do not depend on this
   discrepancy; the native-versus-e2e accuracy difference is descriptive, not
   a clean attribution to the training loss alone.

2. **Initial confidence screen code.** Before the fresh shared-parameter
   confirmation, the statistics-only comparator was tightened to use its own
   fixed-common-noise confidence tube rather than the uniform-angle radius.
   Exact original source files were recovered and verified against the SHA256
   recorded inside `confidence_screen.json`; they are in
   `outputs/code_snapshots/confidence_screen_v1/`. The fresh confirmation uses
   the revised, stronger comparator and entirely new training/query seeds.

3. **Generic verifier extension.** The original `lirpa_plan.json` includes
   CROWN and alpha-CROWN. H8 smoke completed both. It found a 59-second alpha
   computation for about 1% further width reduction versus a .85-second CROWN
   computation. `lirpa_plan_v2.json` records the decision to run CROWN alone
   on the longer-horizon matched cases. This is resource triage; it is not
   evidence that alpha-CROWN or GPU verification cannot solve the problem.

4. **Environment/graph interface repairs.** A NumPy DLL failure in the inherited
   Python3.11 base was fixed only inside the new verifier venv with NumPy2.2.6
   and SciPy1.15.3 wheels. A constant buffer's unintended batch dimension was
   removed before the successful smoke. No auto_LiRPA source was modified.
   The original successful smoke script is retained with its exact recorded
   source hash in `outputs/code_snapshots/lirpa_smoke_v1/`. Later changes add
   CLI/logging support and tensor detachment for metric serialization.

5. **Witness instrumentation.** The feasible-extrema experiment was replayed
   with the same plan, seeds, restart count and step budget to save the actual
   endpoint parameter witnesses and independently check them with NumPy. The
   metric values reproduced; both the prior source/output and the instrumented
   witnesses are retained. No metric-driven optimizer tuning was performed.

6. **Initial Markdown ledger.** The pre-training hypothesis Markdown was
   reconstructed from its earlier tool-read output, rather than pretending an
   independent pre-training cryptographic archive existed. Its archive states
   this limitation. The executable configs/plans and result hashes are separate.
