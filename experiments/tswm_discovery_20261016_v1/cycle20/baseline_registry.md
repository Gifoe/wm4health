# Baseline registry: completed72-fit pilot

Three Markov physics families, each under an invertible12D mixed observation with nine AR nuisance factors and one current visible fresh gain. Each method is independently trained per family,3initializations. This is neither one cross-system pretrained model nor a real-data benchmark.

| Arm | Interface/objective | Scope |
|---|---|---|
| joint |4D shared encoder, joint future-target MSE, variance/covariance penalty | Mathematical JEPA-style control |
| joint_idm | Same+action inverse MSE | Same-protocol inverse control; adds anti-collapse beyond native SMWM |
| ema_idm | EMA detached target+.99 update+inverse | Existing asymmetry control |
| gaussian_idm | Diagonal latent Gaussian NLL+inverse | Simplified probability control, not native Var-JEPA |
| mixture_idm |2head joint-target hard-min MSE+inverse | Omits source MoP's normalization/EMA/router; not native MoP |
| obs_idm |4D input bottleneck predicts full next observation+inverse | Ordinary observation-prediction control |
| fullobs | Raw12D observation MLP, no learned bottleneck | Larger state access, ordinary positive control |
| joint_idm_full |12D learned joint+inverse | Capacity control |

Each world fit uses1500 Adam updates,batch256,lr1e-3,64-wide MLPs. Physical labels are used only after encoder freezing for diagnostic current/successor probes; they never update world encoders. Probe candidates: linear ridge and64-wide MLP,500 updates, validation-selected. ExtraTrees controls are post-pilot exploratory checks on27 frozen encoders, not new independent confirmations.

Post-pilot classical control:18 fits use the same9 full-observation teachers. Top4 input gradient-covariance directions predict either all12 teacher outputs or the learned2D action-response output span; train-only whitening and polynomial ridge of degree1/2/3 select by validation latent error. Physical labels are confined to a final linear diagnostic readout. This is an established compressor with two output-target choices, not a new active-subspace algorithm. Models and raw predictions are in outputs/active_subspace; independent monomial replay and arithmetic passed.

All checkpoints,9split files,72raw prediction files and job stdout/stderr/source snapshots are on the server under outputs/innovation_pilot. Tree checkpoints and predictions are under outputs/strong_probes. Compact outputs and reports are mirrored locally. Do not claim all raw arrays/weights were downloaded locally.

Runtime: existing project Python3.10.20,torch2.8.0+cu128,numpy1.26.4,RTX5090. No environment installation or previous-cycle modification. Exact data seeds,train-only normalization,model seeds and hashes are in the frozen plan/config/registry. Current cycle21 prototype has a new independent cohort and separate protocol; no result is yet claimed here.
