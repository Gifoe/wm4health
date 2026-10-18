# Recovery after the terminal NVIDIA-driver incident

The first fit terminated at 06:14 UTC with Windows access violation 0xc0000005. Windows Error Reporting identifies nvcuda64.dll; the underlying cause is unestablished. Original logs, failed exit receipts and checkpoints remain intact. The original campaign and dependent assessment queue terminated; no free-VAL assessment started.

The last periodic checkpoint is epoch650, with10416 module/proposal updates. All174 inspected tensor values are finite. The saved state includes model/buffers, both optimizers, scheduler, graph assignments, CPU/CUDA/NumPy/Python RNG and both shuffle streams. The exact binary64 temperature is recovered by replaying the original multiply/divide decisions against every logged SA/temp value; no rounded inverse-log approximation is used.

The continuation adapter changes only three entry operations in the source method: restore temperature, initialize structure bookkeeping then restore checkpoint/RNG, and begin the original epoch loop at the saved epoch+1. The entire original epoch-loop body is AST-identical. Losses, proposals, gradients, learning-rate logic, data, batch sizes and checkpoint-selection grid are unchanged.

One critical-path qualification compares a native uninterrupted three-epoch run with continuation from its epoch0 checkpoint. It compares parameters/buffers, both optimizers, scheduler, RNG/shuffle states, structures, annealing statistics and scalar metrics. This is a numeric continuation qualification, not performance evidence. The first adapter attempt rejected the official -1 early-stop sentinel; its versioned correction accepts all nonpositive values, exactly matching the source condition. Preserve both attempts.

After qualification passes, continue the first fit from checkpoint650 through epoch1999, then run the two not-yet-started original seeds under the original algorithm/configuration. Effective completed lineage updates remain32000 per fit. Epoch651 and any partial652 work from the crashed process are discarded; actual physical work is larger and must not be reported as exactly96000 updates overall. Keep all old checkpoints and include the original periodic grid in selection.

Recovery processes set CUDA_LAUNCH_BLOCKING=1 for synchronous diagnostic execution. This is not a verified fix for the driver fault and changes wall-clock behavior. Do not change the shared environment, NVIDIA driver, model, loss, learning rate or evaluation protocol. Stop on another terminal error and diagnose it; never restart because a polling handle merely timed out.

A new campaign and queue use new receipt names. The assessment gate must verify all three complete fits, their successful terminal receipts and weight hashes. Its forecast/control functions and frozen FREE_SIMULATION_PROTOCOL71_V1.md remain unchanged; only the terminal-campaign gate and receipt path are updated. EVAL remains closed and no final research direction is accepted.
