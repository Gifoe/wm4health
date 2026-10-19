# Frozen native chart/cost audit,not a new method experiment

Use the four historical trained WorldTS models NP/PJM xH24/H360,with source revision269426ea6217d7152a244da0e209d922acd46d4f and checkpoint/data/config hashes already recorded in the sealed main audit. Recheck those hashes before inference. No weights,old configs,data or previous experiment files may be changed. Use training contexts only,no test labels or causal-effect interpretation.

For32origins and shifts1,8,24where compatible with the context length,encode overlapping raw patches in the original and updated windows. Exclude the causal-padding boundary so compared raw patches are identical. Measure target and historical-condition state relative difference and cosine. Control: use the original window statistics for both encodings;matching raw patches should then encode identically. Save contexts,statistics,states and outputs needed to replay the comparison. A coordinate-dependent state change alone is not an implementation defect or a forecasting failure.

Profile batch1and8:complete native forecast versus only input statistics plus historical target/condition encoders. Use warm-up and three timing repetitions with40calls each,record synchronized wall latency and CUDA-event timing. The encoder fraction is a descriptive upper bound on forecast speedup from avoiding only those encoders;other computations still need recomputation. Do not present encoder-only time as full cached-world-model inference.

If exact re-encoding is a small portion of the native workload,or an existing fixed/causal-statistic strategy suffices without a demonstrated accuracy penalty,do not introduce a new transport architecture. Large representational drift without practical cost/accuracy consequences does not pass the method motivation gate. This audit does not test all normalization strategies,foundation-model cache designs or actual autoregressive physics simulation.

All numerical work runs on the private5090server. Retain source/input/checkpoint hashes,configs,raw arrays,timing samples and output/weight equality checks. No retraining or public publication.
