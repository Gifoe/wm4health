# Ordinary exact-control cost gate

Frozen before timing results. Native MTS3 `Predict` worker compared with conventional dense augmented-state Gaussian covariance propagation retaining fast/context cross-covariance. This is inference on prescribed weights; neither trained performance nor training memory/throughput is measured.

Native latent_obs_dim d in {1,30,128}; fast state and context each2d; augmented dimension4d. Batch1 and32. Horizon32 within the native slow-window75. Float32/eager CUDA for both. A is a stable dense prescribed matrix (.85I plus a small normalized dense perturbation), C=.1I, Q=.01I, initial fast covariance.05I and context covariance.5I. Native keeps its block-diagonal covariance approximation; full joint reference keeps complete covariance. Timing is a comparison of operators with different retained information, not an equal-arithmetic implementation benchmark.

Warm up4 rollouts;3 repetitions of5 rollouts each; synchronized wall and CUDA event times. Measure allocated GPU memory above the live baseline and peak allocated bytes for one rollout, not reserved allocator bytes. Count configuration/weights in baseline and separate peak increment. Time constructors and matrix assembly outside the measured function for both methods. No torch.compile or custom kernels.

Gate: if ordinary dense augmented inference is already practical at canonical d30 on one5090, do not claim an established need for a new compressed cross-covariance method. Larger dimensions may change scaling but require a real use case and matching actual trained-model evidence. This gate does not establish training cost, parameter learning quality, observation-update correctness, calibrated decoded forecasts or usefulness of the context channel in learned models.
