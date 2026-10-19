# NCDSSM source audit — qualification pending

Author source is [clear-nus/NCDSSM at the pinned commit](https://github.com/clear-nus/NCDSSM/tree/7212bba9616d81c948f507eef9eb1f7d5cf1954e). The source cache manifest records full per-file hashes. The numerical source is unchanged.

`src/ncdssm/models/ncdssm.py` provides complete auxiliary-inference/emission wrappers and LTI/locally-linear/nonlinear state dynamics. Forward returns ELBO/IWELBO and likelihood/regularizer. Recognition produces auxiliary samples from observations/masks; the native state-space filter evaluates their temporal law. This is already a recognition–dynamics interface. A new framework cannot claim that separation itself as missing.

The public constructor asserts `u_dim==0`. Do not silently treat nominal control parameters as an implemented action-conditioned benchmark. Autonomous dynamics is sufficient for the present qualification. A later static system context/modular drift wrapper would be a disclosed new adaptation, not original action support.

Two source-observed sampling details matter for future-law scoring:

1. The outer `NCDSSM.forecast` calls auxiliary inference with one sample per history before asking the base model for multiple future draws. Ensemble members in one such call therefore share an auxiliary-history realization. An estimator requiring iid draws of the fully mixed recognition/forecast law needs an explicit outer-sampling contract; independent repeated histories with one returned future per batch member is an ordinary API-level control.
2. In `src/ncdssm/models/base.py`, `Base.forecast` advances mean/covariance parameters between future times, samples the corresponding predictive state distribution at each time, and retains the propagated parameters for the next step. It does not feed the sampled state into the next transition. Source logic therefore describes marginal-time samples rather than a Markov path draw. A linear numerical test will check timewise moments and cross-time covariance separately. This is a sampling/evaluation distinction, not evidence of bad trained point forecasts, a failure on all tasks, or a new architectural gap.

The ordinary complete comparator should distinguish the native reported forecast API from a joint-path simulator of the fitted generative dynamics, using the same filtered history information and parameters. Preserve the native result, report any adaptation, and do not use a weaker sampling contract to manufacture a proposed-method gain.

`setup.py` declares torch~=1.11 and a Git torchdiffeq dependency; the current server uses torch2.8/Python3.10. Core inference imports a private `torchdiffeq._impl.rk_common.rk4_alt_step_func` symbol. Compatibility and numerical checks are required before calling this implementation operational. No global package downgrade or installation of unrelated physics/data extras is warranted by this source audit.
