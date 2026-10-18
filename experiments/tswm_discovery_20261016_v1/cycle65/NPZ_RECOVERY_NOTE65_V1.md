# Checkpoint Tensor serialization workaround

Isolation did not eliminate all faults. After16 complete fits, nonlinear/seed2023/coverage attempts1 and2 crashed in torch.Tensor._reduce_ex_internal called by torch.save. Their last flushed optimizer steps are2000 and6000; a valid step4000 checkpoint from attempt2 remains. Preserve both fault logs and partial files. The prior two-retry strategy is stopped; a different storage backend is qualified before resuming.

The NPZ backend serializes only numeric arrays plus an explicit JSON tree, with allow_pickle=False. It avoids Tensor pickle reducers entirely. A complete real step4000 model/optimizer/RNG checkpoint roundtrip checked302 tensor fields bitwise. Resume that exact4000 state using the recorded CPU/CUDA/NumPy RNG streams; keep2000/8000 as the only model-selection checkpoints. The remaining MDN uses the same backend. Architecture, losses, seeds, data, optimizer and evaluation are unchanged.

The final archive retains both legacy PT and new NPZ checkpoints, selected through an explicit format-dispatch loader. Validate all weights and metrics under the same isolated runtime. Additional duplicated updates from the two failed coverage attempts are recorded separately (>=4000), besides the original failed cell's8000. This is a serialization-path mitigation, not a proof of the ultimate crash root cause and not a scientific method.
