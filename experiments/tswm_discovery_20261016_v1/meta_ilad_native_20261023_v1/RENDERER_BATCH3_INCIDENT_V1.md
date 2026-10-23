# Preserve a failed native batching check; test the ordinary axis explanation

The original geometry v1 job exited1. Soft alpha, original OBJ and camera checks returned before the full19-frame versus4+4+4+4+3 batching assertion failed with max pixel difference0.9939133525. Those earlier checks were not persisted individually, so the failure receipt alone must not be cited as their independent final proof.

Source inspection shows unqualified `torch.cross` calls on camera arrays of shape[B,3]. The documented legacy default chooses the first dimension of size3: B=3 can therefore change the axis. The native generator renders60-frame sequences, so its batch250 has15000 frames and its normal frame batches never equal3. Do not silently alter its normal renderer.

Geometry v2 retains the original Python/CUDA source. Re-execute geometry and compare19 frames against5+5+5+4 native batches. Separately persist the original3-frame output and compare it with the same frames from the19-frame call. In one scoped, zero-training diagnostic, intercept only otherwise-unspecified `torch.cross` calls to adddim=-1 and compare the3-frame output again. Restore the original function before further work. The prediction is exact agreement with the19-frame reference after this ordinary correction; keep all results if falsified. This control is not a new method and does not establish paper-table parity.
